// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_weights_id = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table_id = tt::CBIndex::c_2;
constexpr uint32_t cb_indices_id = tt::CBIndex::c_3;
constexpr uint32_t cb_output_id = tt::CBIndex::c_16;
constexpr uint32_t cb_local_mask_id = tt::CBIndex::c_4;  // per-chunk slot mask for the reader (see reader)
constexpr uint32_t cb_final_id = tt::CBIndex::c_18;      // residual: reduced + residual tiles (from the compute)

// Fixed CT arg layout for both paths. The dispatch_table / indices metadata
// and accessor args are always emitted by the program factory; when the
// GPT-OSS (weight==0) path is selected, the metadata values are zero and
// the corresponding TensorAccessorArgs reuse the weight tensor's layout so
// the CT arg vector has a deterministic size. use_dispatch_table_skip is
// appended last to guard the runtime code paths that would consult the
// unused slots.
constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_out_tiles = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(3);
constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(4);
constexpr uint32_t dispatch_table_aligned_page_size = get_compile_time_arg_val(5);
constexpr uint32_t indices_page_size = get_compile_time_arg_val(6);
constexpr uint32_t indices_aligned_page_size = get_compile_time_arg_val(7);

constexpr auto weight_accessor_args = TensorAccessorArgs<8>();
constexpr auto output_accessor_args = TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();
constexpr auto dispatch_table_accessor_args =
    TensorAccessorArgs<output_accessor_args.next_compile_time_args_offset()>();
constexpr auto indices_accessor_args =
    TensorAccessorArgs<dispatch_table_accessor_args.next_compile_time_args_offset()>();
constexpr bool use_dispatch_table_skip =
    get_compile_time_arg_val(indices_accessor_args.next_compile_time_args_offset()) != 0;
// output tiles per 32-token row of the whole tensor (emb_dim_out_tiles = this core's column group)
constexpr uint32_t row_tiles = get_compile_time_arg_val(indices_accessor_args.next_compile_time_args_offset() + 1);
// residual: the compute's reduced + residual tiles (c_18, emb_dim_out_tiles per chunk) are written instead of c_16
constexpr bool has_residual = get_compile_time_arg_val(indices_accessor_args.next_compile_time_args_offset() + 2) != 0;
// Per-token weights ([..., num_experts]): a page holds one token's weights; a chunk's 32 pages are read once into
// cb_wtok and each weight is copied into its cb_weights tile by this RISC, instead of one DRAM read per weight.
constexpr bool weights_per_token =
    get_compile_time_arg_val(indices_accessor_args.next_compile_time_args_offset() + 3) != 0;
constexpr uint32_t weights_aligned_page_size =
    get_compile_time_arg_val(indices_accessor_args.next_compile_time_args_offset() + 4);
constexpr uint32_t cb_wtok_id = tt::CBIndex::c_6;

constexpr uint32_t TOKENS_PER_CHUNK = 32;

// Per-token weights: copy token token_idx's weight for slot expert_idx from the chunk's pages (wtok_l1) into the
// weight tile at dst_addr. The compute kernel reads element 0 (scalar broadcast; the zero test reads elements 0-1).
FORCE_INLINE void copy_token_weight(uint32_t dst_addr, uint32_t wtok_l1, uint32_t token_idx, uint32_t expert_idx) {
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_addr);
    const volatile tt_l1_ptr uint16_t* src =
        reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(wtok_l1 + token_idx * weights_aligned_page_size);
    dst[0] = src[expert_idx];
    dst[1] = 0;
}

void kernel_main() {
    // Runtime arg layout: weight_addr, output_addr,
    //   (if dispatch_table_skip: dispatch_table_addr, indices_addr),
    //   token_start_idx, num_chunks, col_tile_start.
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t dispatch_table_addr;
    uint32_t indices_addr;
    uint32_t token_start_idx;
    uint32_t num_chunks;
    if constexpr (use_dispatch_table_skip) {
        dispatch_table_addr = get_arg_val<uint32_t>(2);
        indices_addr = get_arg_val<uint32_t>(3);
        token_start_idx = get_arg_val<uint32_t>(4);
        num_chunks = get_arg_val<uint32_t>(5);
    } else {
        dispatch_table_addr = 0;
        indices_addr = 0;
        token_start_idx = get_arg_val<uint32_t>(2);
        num_chunks = get_arg_val<uint32_t>(3);
    }
    const uint32_t col_tile_start = get_arg_val<uint32_t>(use_dispatch_table_skip ? 6 : 4);

    Noc noc;
    CircularBuffer cb_weights(cb_weights_id);
    CircularBuffer cb_dispatch_table(cb_dispatch_table_id);
    CircularBuffer cb_indices(cb_indices_id);
    CircularBuffer cb_output(cb_output_id);
    CircularBuffer cb_local_mask(cb_local_mask_id);
    CircularBuffer cb_final(cb_final_id);

    const uint32_t weight_tile_size = cb_weights.get_tile_size();
    const uint32_t output_tile_size = cb_output.get_tile_size();

    const auto weight_addrg = TensorAccessor(weight_accessor_args, weight_addr);
    const auto weight_tok_addrg = TensorAccessor(weight_accessor_args, weight_addr, weights_aligned_page_size);
    uint32_t wtok_l1 = 0;
    if constexpr (weights_per_token) {
        CircularBuffer cb_wtok(cb_wtok_id);
        wtok_l1 = cb_wtok.get_write_ptr();
    }
    const auto output_addrg = TensorAccessor(output_accessor_args, output_addr);

    uint32_t dispatch_table_write_addr = 0;

    if constexpr (use_dispatch_table_skip) {
        const auto dispatch_table_addrg =
            TensorAccessor(dispatch_table_accessor_args, dispatch_table_addr, dispatch_table_page_size);

        // Pre-load dispatch table into CB (c_2) — read once, used by compute for all chunks
        cb_dispatch_table.reserve_back(dispatch_table_num_pages);
        dispatch_table_write_addr = cb_dispatch_table.get_write_ptr();
        for (uint32_t i = 0; i < dispatch_table_num_pages; i++) {
            noc.async_read(
                dispatch_table_addrg,
                cb_dispatch_table,
                dispatch_table_aligned_page_size,
                {.page_id = i},
                {.offset_bytes = i * dispatch_table_aligned_page_size});
        }
        noc.async_read_barrier();
        cb_dispatch_table.push_back(dispatch_table_num_pages);
    }

    constexpr uint32_t cb_output_tiles = emb_dim_cb_tiles * TOKENS_PER_CHUNK;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Load indices for this chunk into CB (c_3) — one chunk at a time to save L1
        uint32_t indices_write_addr = 0;
        if constexpr (use_dispatch_table_skip) {
            const auto indices_addrg = TensorAccessor(indices_accessor_args, indices_addr, indices_page_size);
            cb_indices.reserve_back(TOKENS_PER_CHUNK);
            indices_write_addr = cb_indices.get_write_ptr();
            for (uint32_t i = 0; i < TOKENS_PER_CHUNK; i++) {
                noc.async_read(
                    indices_addrg,
                    cb_indices,
                    indices_aligned_page_size,
                    {.page_id = token_start_idx + i},
                    {.offset_bytes = i * indices_aligned_page_size});
            }
            noc.async_read_barrier();
            // Indices stay as uint16 in the CB — the compute kernel reads them
            // directly via read_tile_value_uint16.
            cb_indices.push_back(TOKENS_PER_CHUNK);

            // Slot mask for the reader, with the compute's skip rule: local slots are read; a token with no local
            // expert gets zeros in its last slot (the accumulator-init multiply); every other slot is skipped.
            cb_local_mask.reserve_back(1);
            volatile tt_l1_ptr uint8_t* mask =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cb_local_mask.get_write_ptr());
            tt_l1_ptr int32_t* dt = reinterpret_cast<tt_l1_ptr int32_t*>(dispatch_table_write_addr);
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
                tt_l1_ptr uint16_t* ti =
                    reinterpret_cast<tt_l1_ptr uint16_t*>(indices_write_addr + t * indices_aligned_page_size);
                bool any = false;
                for (uint32_t k = 0; k < num_experts; ++k) {
                    const bool local = dt[ti[k]] != -1;
                    mask[t * num_experts + k] = local ? 1 : 0;
                    any = any || local;
                }
                if (!any) {
                    mask[t * num_experts + num_experts - 1] = 2;
                }
            }
            cb_local_mask.push_back(1);
        }

        if constexpr (weights_per_token) {
            for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
                noc.async_read(
                    weight_tok_addrg,
                    CoreLocalMem<uint32_t>(wtok_l1 + i * weights_aligned_page_size),
                    weights_aligned_page_size,
                    {.page_id = token_start_idx + i},
                    {});
            }
            noc.async_read_barrier();
        }

        // Phase 1: Stream one weight per expert per token for this chunk
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            bool has_local = true;
            if constexpr (use_dispatch_table_skip) {
                tt_l1_ptr int32_t* dispatch_table = reinterpret_cast<tt_l1_ptr int32_t*>(dispatch_table_write_addr);
                // Indices stay as uint16 in the CB — no in-place expansion.
                tt_l1_ptr uint16_t* token_indices =
                    reinterpret_cast<tt_l1_ptr uint16_t*>(indices_write_addr + token_idx * indices_aligned_page_size);
                has_local = false;
                for (uint32_t k = 0; k < num_experts; ++k) {
                    if (dispatch_table[token_indices[k]] != -1) {
                        has_local = true;
                        break;
                    }
                }
            }

            // one batch per token: read only the weights the compute uses (local slots; the compute skips the
            // rest without looking at them), one barrier per token
            cb_weights.reserve_back(num_experts);
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                const uint32_t slot_off = expert_idx * weight_tile_size;
                const uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
                if constexpr (use_dispatch_table_skip) {
                    const bool is_last = (expert_idx == num_experts - 1);
                    tt_l1_ptr int32_t* dispatch_table = reinterpret_cast<tt_l1_ptr int32_t*>(dispatch_table_write_addr);
                    tt_l1_ptr uint16_t* token_indices = reinterpret_cast<tt_l1_ptr uint16_t*>(
                        indices_write_addr + token_idx * indices_aligned_page_size);
                    if (!has_local && is_last) {
                        // No local experts for this token — zero the weight tile so compute's
                        // must_zero_init multiply produces zeros regardless of combine_output.
                        volatile tt_l1_ptr uint32_t* ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_weights.get_write_ptr() + slot_off);
                        for (uint32_t w = 0; w < weight_tile_size / sizeof(uint32_t); w++) {
                            ptr[w] = 0;
                        }
                    } else if (dispatch_table[token_indices[expert_idx]] != -1) {
                        if constexpr (weights_per_token) {
                            copy_token_weight(cb_weights.get_write_ptr() + slot_off, wtok_l1, token_idx, expert_idx);
                        } else {
                            noc.async_read(
                                weight_addrg,
                                cb_weights,
                                weight_tile_size,
                                {.page_id = weight_page_idx},
                                {.offset_bytes = slot_off});
                        }
                    }
                } else if constexpr (weights_per_token) {
                    copy_token_weight(cb_weights.get_write_ptr() + slot_off, wtok_l1, token_idx, expert_idx);
                } else {
                    noc.async_read(
                        weight_addrg,
                        cb_weights,
                        weight_tile_size,
                        {.page_id = weight_page_idx},
                        {.offset_bytes = slot_off});
                }
            }
            noc.async_read_barrier();
            cb_weights.push_back(num_experts);
        }

        // Phase 2: Write output tiles after compute finishes this chunk.
        // The output CB holds TOKENS_PER_CHUNK * emb_dim_cb_tiles tile-sized pages
        // (including padding when emb_dim is not 1024-aligned); only the first
        // emb_dim_out_tiles of them hold real data for this 32-token block.
        // residual: the compute's reduced + residual tiles (c_18, emb_dim_out_tiles) instead of c_16.
        CircularBuffer& cb_src = has_residual ? cb_final : cb_output;
        constexpr uint32_t src_tiles = has_residual ? emb_dim_out_tiles : cb_output_tiles;
        cb_src.wait_front(src_tiles);

        uint32_t tile_row = token_start_idx / TOKENS_PER_CHUNK;
        uint32_t start_tile_idx = tile_row * row_tiles + col_tile_start;

        for (uint32_t tile_idx = 0; tile_idx < emb_dim_out_tiles; ++tile_idx) {
            noc.async_write(
                cb_src,
                output_addrg,
                output_tile_size,
                {.offset_bytes = tile_idx * output_tile_size},
                {.page_id = start_tile_idx + tile_idx});
        }

        noc.async_write_barrier();

        cb_src.pop_front(src_tiles);

        token_start_idx += TOKENS_PER_CHUNK;
    }
}
