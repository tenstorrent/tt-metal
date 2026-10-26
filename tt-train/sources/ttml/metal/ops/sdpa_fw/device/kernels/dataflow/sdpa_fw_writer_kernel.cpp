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
    const uint32_t work_list_arg_idx = runtime_args_counter;  // then the work list

    constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
    constexpr uint32_t cb_output = tt::CBIndex::c_15;

    constexpr uint32_t vWt = get_compile_time_arg_val(0);         // number of tiles in output/value inner dimension
    constexpr uint32_t Ht = get_compile_time_arg_val(1);          // number of tiles in sequence dimension
    constexpr uint32_t q_heads = get_compile_time_arg_val(2);     // num of heads in query
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);  // query tile-rows per Q chunk
    constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;

    const uint32_t tile_bytes = get_tile_size(cb_output);

    constexpr auto output_args = TensorAccessorArgs<4>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr);

    [[maybe_unused]] constexpr auto intermediates_args =
        TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
#ifdef RETURN_INTERMEDIATES
    const uint32_t interm_tile_bytes = get_tile_size(cb_intermediates);
    const auto intermediates_addr_generator = TensorAccessor(intermediates_args, intermediates_addr, interm_tile_bytes);
#endif

    constexpr uint32_t onetile = 1U;

    // Generate tiles that compute kernel needs BEFORE reader starts pushing data
    // This allows reader to start DRAM reads immediately while writer generates these tiles
    constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;

    constexpr uint16_t one = 0x00003F80;                          // (bfloat16)1.0 -> uint16_t
    generate_tile_with_bfloat16_value(cb_reduction_scaler, one);  // tile with 1.0 for reduction
    generate_matmul_row_reduce_tile(cb_matmul_reduce);            // tile for matmul row reduce

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

    constexpr uint32_t out_tiles_per_chunk = Sq_chunk_t * vWt;
    constexpr uint32_t kIntermediateTilesPerRow = 1U;

    // One Q chunk: Sq_chunk_t consecutive output rows, contiguous in DRAM.
    auto write_q_chunk = [&](const uint32_t global_q_chunk_idx) {
        const uint32_t seq_idx = global_q_chunk_idx / q_num_chunks;  // batch * q_heads + head
        const uint32_t first_row_tile = (global_q_chunk_idx % q_num_chunks) * Sq_chunk_t;
        const uint32_t global_first_row = seq_idx * Ht + first_row_tile;

        // -------- Output: (B, H, S, D), heads NOT fused --------
        // Linear index for [B, H, S, D]: ((b * q_heads + h) * Ht + s_tile) * vWt + col
        write_tiles_by_row(
            cb_output,
            output_addr_generator,
            global_first_row * vWt,
            out_tiles_per_chunk,
            tile_bytes,
            out_tiles_per_chunk);

#ifdef RETURN_INTERMEDIATES
        // -------- Intermediates: (B, qNH, S, 32) = 1 FP32 tile per row: logsumexp --------
        constexpr uint32_t interm_tiles = Sq_chunk_t * kIntermediateTilesPerRow;
        cb_wait_front(cb_intermediates, interm_tiles);
        uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);
        for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
            noc_async_write_page(global_first_row + r, intermediates_addr_generator, l1_intermediates_read_addr);
            l1_intermediates_read_addr += interm_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_intermediates, interm_tiles);
#endif
    };

    for_each_work_item(work_list_arg_idx, write_q_chunk);
}
