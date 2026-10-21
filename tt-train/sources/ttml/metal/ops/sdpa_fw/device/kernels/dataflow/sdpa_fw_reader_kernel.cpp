// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t query_address = get_arg_val<uint32_t>(runtime_args_counter++);  // query buffer address
    uint32_t key_address = get_arg_val<uint32_t>(runtime_args_counter++);    // key buffer address
    uint32_t value_address = get_arg_val<uint32_t>(runtime_args_counter++);  // value buffer address
    uint32_t mask_address = get_arg_val<uint32_t>(runtime_args_counter++);   // mask buffer address
    uint32_t gate_address = get_arg_val<uint32_t>(runtime_args_counter++);   // gate buffer address
    // Work units handed to this core: Q chunks in standard mode, Q chunk pairs in balanced mode.
    uint32_t num_units = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_unit = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
#endif
#ifdef HAS_GATE
    constexpr uint32_t cb_gate = tt::CBIndex::c_16;
#endif

    constexpr uint32_t qWt = get_compile_time_arg_val(0);      // num tile in inner dim in query/key (d_qk/TILE_W)
    constexpr uint32_t vWt = get_compile_time_arg_val(1);      // num tile in inner dim in value (d_v/TILE_W)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);       // (S / TILE_H)
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);  // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // num of heads per group
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);       // K/V tiles per inner-loop step
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);       // query tile-rows per Q chunk
    constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
    [[maybe_unused]] constexpr uint32_t pairs_per_seq = q_num_chunks / 2;
    constexpr auto query_args = TensorAccessorArgs<7>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
#ifdef HAS_GATE
    constexpr auto gate_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
#endif

    const uint32_t tile_bytes = get_tile_size(cb_query);

    const auto query_address_generator = TensorAccessor(query_args, query_address);
    const auto key_address_generator = TensorAccessor(key_args, key_address);
    const auto value_address_generator = TensorAccessor(value_args, value_address);

#ifdef USE_ATTN_MASK
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);
#endif
#ifdef HAS_GATE
    const auto gate_address_generator = TensorAccessor(gate_args, gate_address);
#endif

    // Note: Tile generation (reduction_scaler, matmul_reduce, causal_mask) moved to writer kernel
    // This allows reader to start DRAM reads immediately

    const uint32_t num_of_groups = q_heads / heads_per_group;

    // Stage one Q chunk (Sq_chunk_t consecutive query tile-rows of one batch/head) and the
    // K/V stream it attends to. K and V go in Sk_chunk_t-tile steps (one DMA + one push per
    // step); every step is read once and serves all Sq_chunk_t rows of the chunk.
    //
    // For causal the K-tile count is rounded UP to the next Sk_chunk_t boundary past the
    // chunk's last row so the diagonal band is fully resident before compute processes it.
    // Ht is a multiple of both chunk sizes (program factory), so the round-up never reads OOB.
    auto read_q_chunk = [&](const uint32_t global_q_chunk_idx) {
        const uint32_t seq_idx = global_q_chunk_idx / q_num_chunks;  // batch * q_heads + head
        const uint32_t first_row_tile = (global_q_chunk_idx % q_num_chunks) * Sq_chunk_t;
        const uint32_t global_first_row = seq_idx * Ht + first_row_tile;

        // Q rows of the chunk are consecutive in DRAM: one read of Sq_chunk_t * qWt tiles.
        read_tiles_by_row(
            cb_query, query_address_generator, global_first_row * qWt, Sq_chunk_t * qWt, tile_bytes, Sq_chunk_t * qWt);

        const uint32_t q_head_idx = seq_idx % q_heads;
        const uint32_t batch_idx = seq_idx / q_heads;
        const uint32_t kv_group_idx = q_head_idx / heads_per_group;
        const uint32_t key_offset = (batch_idx * num_of_groups + kv_group_idx) * qWt * Ht;
        const uint32_t value_offset = (batch_idx * num_of_groups + kv_group_idx) * vWt * Ht;

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
        const uint32_t num_kv_tiles_to_read = round_up(first_row_tile + Sq_chunk_t, Sk_chunk_t);
#else
        const uint32_t num_kv_tiles_to_read = Ht;
#endif
        const uint32_t num_kv_steps = num_kv_tiles_to_read / Sk_chunk_t;

        for (uint32_t k_step = 0; k_step < num_kv_steps; ++k_step) {
            const uint32_t h_start = k_step * Sk_chunk_t;
            const uint32_t key_start_idx = key_offset + h_start * qWt;
            // Lay K out col-major in cb_key (feat outer, seq inner) so `matmul_block`
            // with transpose=1 can step in1_idx by +Sk_chunk_t per feature, walking the
            // contraction direction contiguously. K source in DRAM is (Sk_chunk_t × qWt)
            // row-major; the helper writes tile (n, feat) at CB pos feat*Sk_chunk_t + n.
            read_tile_block_transposed(cb_key, key_address_generator, key_start_idx, Sk_chunk_t, qWt, tile_bytes);

#ifdef USE_ATTN_MASK
            // Mask is (1, 1, S, S): the Sq_chunk_t x Sk_chunk_t tiles of this step, row-major
            // (row r, K tile n at r * Sk_chunk_t + n), staged in one DMA + one push. Each
            // row's Sk_chunk_t tiles are contiguous along the K axis.
            {
                constexpr uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
                cb_reserve_back(cb_attn_mask, mask_tiles);
                uint32_t l1_addr = get_write_ptr(cb_attn_mask);
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    uint32_t src_tile_id = (first_row_tile + r) * Ht + h_start;
                    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
                        noc_async_read_page(src_tile_id, mask_address_generator, l1_addr);
                        src_tile_id += 1U;
                        l1_addr += tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_attn_mask, mask_tiles);
            }
#endif
            // Note: For CAUSAL_MASK, the two reusable mask tiles are generated once by the writer.
            const uint32_t value_start_idx = value_offset + h_start * vWt;
            read_tiles_by_row(
                cb_value, value_address_generator, value_start_idx, Sk_chunk_t * vWt, tile_bytes, Sk_chunk_t * vWt);
        }
#ifdef HAS_GATE
        read_tiles_by_row(
            cb_gate, gate_address_generator, global_first_row * vWt, Sq_chunk_t * vWt, tile_bytes, Sq_chunk_t * vWt);
#endif
    };

#ifdef BALANCED_PARALLELISM
    // Balanced parallelism mode: each unit is a pair of Q chunks of one sequence, the
    // light chunk (early in the sequence, few K/V steps) with the heavy chunk (late, many
    // steps), so every core reads roughly the same amount of data. Heavy first (LPT).
    for (uint32_t p = 0; p < num_units; ++p) {
        const uint32_t global_pair_idx = start_unit + p;
        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;
        read_q_chunk(seq_idx * q_num_chunks + (q_num_chunks - 1U - pair_in_seq));
        read_q_chunk(seq_idx * q_num_chunks + pair_in_seq);
    }
#else
    // Standard mode: consecutive Q chunks.
    for (uint32_t i = 0; i < num_units; ++i) {
        read_q_chunk(start_unit + i);
    }
#endif
}
