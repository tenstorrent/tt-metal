// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compile_time_args.h>

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/sdpa_fw_work_list.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t scratch_addr = get_arg_val<uint32_t>(runtime_args_counter++);  // partials scratch (split units)
    const uint32_t work_list_arg_idx = runtime_args_counter;                // then the work list

    constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
    constexpr uint32_t cb_output = tt::CBIndex::c_15;
#ifdef SPLIT_UNITS
    constexpr uint32_t cb_partial_out = tt::CBIndex::c_18;
#endif

    constexpr uint32_t vWt = get_compile_time_arg_val(0);             // number of tiles in output/value inner dimension
    constexpr uint32_t Ht = get_compile_time_arg_val(1);              // number of tiles in sequence dimension
    constexpr uint32_t q_heads = get_compile_time_arg_val(2);         // num of heads in query
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);      // query tile-rows per Q chunk
    constexpr uint32_t heads_per_item = get_compile_time_arg_val(4);  // Q heads stacked in one item
    // 5..8: the four semaphore ids (a sender bumps its merger's)
    constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
    constexpr uint32_t rows = Sq_chunk_t * heads_per_item;  // tile-rows of one item

    const uint32_t tile_bytes = get_tile_size(cb_output);

    constexpr auto output_args = TensorAccessorArgs<9>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr);

    [[maybe_unused]] constexpr auto intermediates_args =
        TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
#ifdef RETURN_INTERMEDIATES
    const uint32_t interm_tile_bytes = get_tile_size(cb_intermediates);
    const auto intermediates_addr_generator = TensorAccessor(intermediates_args, intermediates_addr, interm_tile_bytes);
#endif
    [[maybe_unused]] constexpr auto scratch_args =
        TensorAccessorArgs<intermediates_args.next_compile_time_args_offset()>();
#ifdef SPLIT_UNITS
    constexpr uint32_t partial_row_tiles = vWt + 2U;
    const uint32_t partial_tile_bytes = get_tile_size(cb_partial_out);
    const auto scratch_addr_generator = TensorAccessor(scratch_args, scratch_addr, partial_tile_bytes);
#endif

    constexpr uint32_t onetile = 1U;

    // Generate tiles that compute kernel needs BEFORE reader starts pushing data
    // This allows reader to start DRAM reads immediately while writer generates these tiles
    constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;

    generate_tile_with_bfloat16_value(cb_reduction_scaler, BF16_ONE_BITS);  // tile with 1.0 for reduction
    generate_matmul_row_reduce_tile(cb_matmul_reduce);                      // tile for matmul row reduce

#ifdef CAUSAL_MASK
    // Emit *pre-transformed* mask tiles so the compute kernel can stamp them directly
    // onto QK^T scores via the packer's L1-accumulate path (TTNN's
    // `apply_causal_mask_lightweight` pattern). The mask values are added in FP32 in L1 by
    // the packer's read-modify-write — score never leaves FP32, no DST→SRC truncation.
    //
    //   tile[0] = causal-diagonal: 0.0 on/below diagonal (kept), -1e9 above (masked).
    //             Applied on the diagonal K tile of every row.
    //   tile[1] = all -1e9: applied on K tiles strictly past the diagonal inside the
    //             diagonal chunk when Sk_chunk_t > 1.
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
    generate_pretransformed_causal_mask_tile(cb_attn_mask);
    generate_tile_with_bfloat16_value(cb_attn_mask, BF16_NEG_LARGE_BITS);
#endif

    // One item: rows stacked head after head in the CBs, each head's Sq_chunk_t rows contiguous in DRAM.
    auto write_q_chunk = [&](const uint32_t item) {
        const auto w = decode_work_item<q_num_chunks, Sq_chunk_t, q_heads, heads_per_item>(item);

        // -------- Output: (B, H, S, D), heads NOT fused --------
        // Linear index for [B, H, S, D]: ((b * q_heads + h) * Ht + s_tile) * vWt + col
        constexpr uint32_t out_tiles_per_head = Sq_chunk_t * vWt;
        cb_wait_front(cb_output, rows * vWt);
        uint32_t l1_read_addr = get_read_ptr(cb_output);
        for (uint32_t h = 0; h < heads_per_item; ++h) {
            const uint32_t start_idx = ((w.batch * q_heads + w.head0 + h) * Ht + w.first_row_tile) * vWt;
            for (uint32_t t = 0; t < out_tiles_per_head; ++t) {
                noc_async_write_page(start_idx + t, output_addr_generator, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, rows * vWt);

#ifdef RETURN_INTERMEDIATES
        // -------- Intermediates: (B, qNH, S, 32) = 1 FP32 tile per row: logsumexp --------
        cb_wait_front(cb_intermediates, rows);
        uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);
        for (uint32_t h = 0; h < heads_per_item; ++h) {
            const uint32_t first_page = (w.batch * q_heads + w.head0 + h) * Ht + w.first_row_tile;
            for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                noc_async_write_page(first_page + r, intermediates_addr_generator, l1_intermediates_read_addr);
                l1_intermediates_read_addr += interm_tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_intermediates, rows);
#endif
    };

    for_each_work_unit(work_list_arg_idx, [&](const WorkUnit& unit) {
#ifdef SPLIT_UNITS
        if (unit.role == kUnitSender) {
            // The partial state goes to its scratch slot; once it has landed, the merger's semaphore is bumped.
            const uint32_t sem_ids[4] = {
                get_compile_time_arg_val(5),
                get_compile_time_arg_val(6),
                get_compile_time_arg_val(7),
                get_compile_time_arg_val(8)};
            for (uint32_t r = 0; r < rows; ++r) {
                write_tiles_by_row(
                    cb_partial_out,
                    scratch_addr_generator,
                    (unit.slot * rows + r) * partial_row_tiles,
                    partial_row_tiles,
                    partial_tile_bytes,
                    partial_row_tiles);
            }
            noc_semaphore_inc(get_noc_addr(unit.merger_x, unit.merger_y, get_semaphore(sem_ids[unit.sem_idx])), 1);
            noc_async_atomic_barrier();  // the kernel must not end with the increment in flight
            return;
        }
#endif
        for (uint32_t item = unit.item; item < unit.item + unit.num_items; ++item) {
            write_q_chunk(item);
        }
    });
}
