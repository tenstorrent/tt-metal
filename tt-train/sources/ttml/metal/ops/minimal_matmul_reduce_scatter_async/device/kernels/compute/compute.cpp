// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"
#include "tt-train/sources/ttml/metal/ops/minimal_matmul_reduce_scatter_async/device/kernels/minimal_matmul_reduce_scatter_async_utils.hpp"

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

void add_and_pack_block(
    uint32_t in_cb,
    uint32_t incoming_cb,
    uint32_t out_cb,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t rows,
    uint32_t cols) {
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_incoming(incoming_cb);
    reconfig_data_format(in_cb, incoming_cb);
    pack_reconfig_data_format(out_cb);
    add_init(in_cb, incoming_cb);
    cb_incoming.wait_front(M_block_tiles * N_block_tiles);

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        if (m < rows) {
            for (uint32_t n = 0; n < cols; n++) {
                const uint32_t tile = (m * N_block_tiles) + n;
                tile_regs_acquire();
                add_tiles(in_cb, incoming_cb, tile, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, out_cb, n);
                tile_regs_release();
            }
        }
        cb_out.push_back(N_block_tiles);
    }
    cb_incoming.pop_front(M_block_tiles * N_block_tiles);
}

void transpose_in0_block(uint32_t in_cb, uint32_t out_cb, uint32_t in1_cb, uint32_t tiles, uint32_t block_tiles) {
    constexpr uint32_t tiles_per_dst = 4;
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    PACK((llk_pack_reconfig_l1_acc(0)));
    reconfig_data_format_srca(in1_cb, in_cb);
    pack_reconfig_data_format(out_cb);
    transpose_init(in_cb);
    cb_in.wait_front(block_tiles);
    cb_out.reserve_back(block_tiles);
    for (uint32_t first = 0; first < tiles; first += tiles_per_dst) {
        const uint32_t count = std::min(tiles_per_dst, tiles - first);
        tile_regs_acquire();
        for (uint32_t i = 0; i < count; i++) {
            transpose_tile(in_cb, first + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < count; i++) {
            pack_tile<true>(i, out_cb, first + i);
        }
        tile_regs_release();
    }
    cb_out.push_back(block_tiles);
    cb_in.pop_front(block_tiles);
    reconfig_data_format_srca(in_cb, in1_cb);
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
    constexpr uint32_t M_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t m_piece = get_compile_time_arg_val(8);
    constexpr uint32_t n_width = get_compile_time_arg_val(9);
    constexpr uint32_t ring_size = get_compile_time_arg_val(10);
    constexpr bool transpose_b = get_compile_time_arg_val(11) == 1;
    constexpr bool transposes_in0 = get_compile_time_arg_val(12) == 1;

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;
    constexpr uint32_t incoming_cb = tt::CBIndex::c_5;
    constexpr uint32_t mm_in0_cb = transposes_in0 ? tt::CBIndex::c_7 : in0_cb;

    CircularBuffer cb_in0(mm_in0_cb);
    CircularBuffer cb_transposed_ready(tt::CBIndex::c_8);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_intermediate(intermediate_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);
    matmul_init(in0_cb, in1_cb, transpose_b);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    for (uint32_t step = 0; step < ring_size; step++) {
        for (uint32_t nb = 0; nb < N_blocks; nb++) {
            const uint32_t current_N_block_tiles = mmrs::n_block(nb, n_width, N_blocks).size;
            const uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            for (uint32_t mb = 0; mb < M_blocks; mb++) {
                const uint32_t current_M_block_tiles = mmrs::balanced_span(mb, m_piece, M_blocks).size;
                const uint32_t current_subblock_h = std::min(current_M_block_tiles, subblock_h);

                reconfig_data_format(in1_cb, mm_in0_cb);
                pack_reconfig_data_format(intermediate_cb);
                matmul_block_init(
                    mm_in0_cb, in1_cb, transpose_b, current_subblock_w, current_subblock_h, K_block_tiles);
                cb_intermediate.reserve_back(out_block_num_tiles);
                for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                    if constexpr (transposes_in0) {
                        cb_transposed_ready.reserve_back(1);
                        transpose_in0_block(
                            in0_cb, mm_in0_cb, in1_cb, current_M_block_tiles * K_block_tiles, in0_block_num_tiles);
                        cb_transposed_ready.push_back(1);
                        pack_reconfig_data_format(intermediate_cb);
                        matmul_block_init(
                            mm_in0_cb, in1_cb, transpose_b, current_subblock_w, current_subblock_h, K_block_tiles);
                        if (k_block > 0) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }
                    }
                    cb_in0.wait_front(in0_block_num_tiles);
                    cb_in1.wait_front(in1_block_num_tiles);

                    matmul_blocks<transpose_b>(
                        mm_in0_cb,
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
                if (step == 0) {
                    copy_and_pack_block(
                        intermediate_cb,
                        out_cb,
                        M_block_tiles,
                        N_block_tiles,
                        current_M_block_tiles,
                        current_N_block_tiles);
                } else {
                    add_and_pack_block(
                        intermediate_cb,
                        incoming_cb,
                        out_cb,
                        M_block_tiles,
                        N_block_tiles,
                        current_M_block_tiles,
                        current_N_block_tiles);
                }
                cb_intermediate.pop_front(out_block_num_tiles);
            }
        }
    }
}
