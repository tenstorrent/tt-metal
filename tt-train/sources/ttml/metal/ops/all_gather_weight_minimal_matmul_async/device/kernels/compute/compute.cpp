// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tt-train/sources/ttml/metal/ops/all_gather_weight_minimal_matmul_async/device/kernels/all_gather_weight_minimal_matmul_async_utils.hpp"

// Copies only the real tiles of a partial block
void copy_and_pack_block(
    uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles, uint32_t rows, uint32_t cols) {
    CircularBuffer cb_out(out_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(in_cb);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        if (m < rows) {
            for (uint32_t n = 0; n < cols; n++) {
                tile_regs_acquire();
                tile_regs_wait();
                copy_tile(in_cb, (m * N_block_tiles) + n, 0);
                pack_tile<true>(0, out_cb, n);
                tile_regs_commit();
                tile_regs_release();
            }
        }
        cb_out.push_back(N_block_tiles);
    }
}

template <bool transpose_b>
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
        const uint32_t rows = std::min(subblock_h, M_block_tiles - M_start);
        // A short last subblock gets its own init instead of padding the block.
        if (rows != subblock_h) {
            matmul_block_init(in0_cb, in1_cb, transpose_b, subblock_w, rows, K_block_tiles);
        }
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();

            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, transpose_b, subblock_w, rows, K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < rows; h++) {
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t out_tile_id = ((M_start + h) * full_N_block_tiles) + N_start + w;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        if (rows != subblock_h) {
            matmul_block_init(in0_cb, in1_cb, transpose_b, subblock_w, subblock_h, K_block_tiles);
        }
        in0_index_offset += rows * K_block_tiles;
    }
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t piece_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t ring_size = get_compile_time_arg_val(9);
    constexpr bool transpose_b = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t hops_before = get_compile_time_arg_val(11);
    // N blocks are cut per hop region and differ in width, so compute needs the stripe layout too.
    using Blocks = StripeBlocks<N_block_tiles, piece_tiles, ring_size, hops_before>;

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_intermediate(intermediate_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);
    matmul_init(in0_cb, in1_cb, transpose_b);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    // N-outer: every M block of an N block runs before the next one, so remote stripes are needed as late as possible.
    for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
        uint32_t current_N_block_tiles = Blocks::span(n_block_iter).width;
        uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
            uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
            uint32_t current_M_block_tiles = m_tile_end - m_tile;
            uint32_t current_subblock_h = std::min(current_M_block_tiles, subblock_h);

            reconfig_data_format(in1_cb, in0_cb);  // the copy-out left the unpackers set up for itself
            pack_reconfig_data_format(intermediate_cb);
            matmul_block_init(in0_cb, in1_cb, transpose_b, current_subblock_w, current_subblock_h, K_block_tiles);
            cb_intermediate.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                cb_in0.wait_front(in0_block_num_tiles);
                cb_in1.wait_front(in1_block_num_tiles);

                matmul_blocks<transpose_b>(
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
                if (k_block == 0) {  // later K blocks accumulate in L1
                    pack_reconfig_l1_acc(1);
                }
            }

            cb_intermediate.push_back(out_block_num_tiles);
            pack_reconfig_l1_acc(0);

            // Partials are in the accumulation format; copy them out in the output format.
            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
            copy_and_pack_block(
                intermediate_cb, out_cb, M_block_tiles, N_block_tiles, current_M_block_tiles, current_N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
        }
    }
}
