// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qwen_ag_matmul compute: the stock minimal_matmul compute kernel
// (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp) reduced to the path the
// context-parallel prefill runs (no bias / activation / ternary / SwiGLU, one N block per core), with the M blocks
// taken from an explicit per-core list (runtime args) instead of a contiguous M range. Every block is computed with
// the same calls in the same order as the stock kernel computes a block of that height, so the bits do not change.
//
// CT: 0 K_num_blocks, 1 M_block_tiles, 2 K_block_tiles, 3 N_block_tiles, 4 subblock_h, 5 subblock_w,
//     6 N_tiles_per_core, 7 list_offset (common arg index of the block lists)
// Common RT: ..., at list_offset: GX list offsets, then the lists (per grid column: num_blocks, num_blocks x
// (m_start_tile, rows))

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void copy_and_pack_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(in_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        cb_out.push_back(N_block_tiles);
    }
}

void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;

    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

void kernel_main() {
#ifdef PROBE_NOMM
    return;
#endif
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t list_offset = get_compile_time_arg_val(7);

    const uint32_t cx = get_absolute_logical_x();
    const uint32_t N_start_tile = 0;
    const uint32_t N_end_tile = N_tiles_per_core;
    const uint32_t list_at = get_common_arg_val<uint32_t>(list_offset + cx);
    const uint32_t num_blocks = get_common_arg_val<uint32_t>(list_at);
    const uint32_t blocks_idx = list_at + 1;

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_intermediate(intermediate_cb);

    // compute_kernel_hw_startup must be the first compute API call (before SFPU/op inits).
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);

    matmul_init(in0_cb, in1_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    const uint32_t n_tile_end = std::min(N_start_tile + N_block_tiles, N_end_tile);
    const uint32_t current_N_block_tiles = n_tile_end - N_start_tile;
    const uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t rows = get_common_arg_val<uint32_t>(blocks_idx + 2 * b + 1);
        const uint32_t current_M_block_tiles = rows;
        // a zero-row (padding) block computes nothing; the init still needs a non-zero rt_dim
        const uint32_t current_subblock_h = rows == 0 ? subblock_h : std::min(current_M_block_tiles, subblock_h);

        reconfig_data_format(in1_cb, in0_cb);
        pack_reconfig_data_format(intermediate_cb);
        matmul_block_init(
            in0_cb,
            in1_cb,
            false /*transpose*/,
            current_subblock_w /*ct_dim*/,
            current_subblock_h /*rt_dim*/,
            K_block_tiles /*kt_dim*/);
        // Accumulation buffer
        cb_intermediate.reserve_back(out_block_num_tiles);
        for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
            cb_in0.wait_front(in0_block_num_tiles);
            cb_in1.wait_front(in1_block_num_tiles);

            matmul_blocks(
                in0_cb,
                in1_cb,
                intermediate_cb,
                current_M_block_tiles,
                current_N_block_tiles,
                N_block_tiles,
                K_block_tiles,
                current_subblock_h,
                current_subblock_w);

            cb_in0.pop_front(in0_block_num_tiles);
            cb_in1.pop_front(in1_block_num_tiles);
            if (k_block == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
        }

        cb_intermediate.push_back(out_block_num_tiles);
        PACK((llk_pack_reconfig_l1_acc(0)));

        cb_out.reserve_back(out_block_num_tiles);
        cb_intermediate.wait_front(out_block_num_tiles);
        copy_and_pack_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
        cb_intermediate.pop_front(out_block_num_tiles);
    }
}
