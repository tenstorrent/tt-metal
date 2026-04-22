// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Matmul compute kernel — runs on TRISC (all 3 TRISC processors).
// Computes M_block x N_block output blocks for the Mpc×Mpc gram matmul.
// Output streamed per (m_sub, n_sub) block:
//   REDUCE_SENDER/TRANSPOSE: matmul → c_2, pack c_2 → c_5 (row-major), DM sends c_5 to partner
//   REDUCE_ACCUMULATOR: matmul → c_2, add own c_2(FP32) + partner's c_5(BF16) → c_6

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

constexpr uint32_t kDstAccIdx = 0;    // DST tile index for accumulation
constexpr bool kTransposeIn1 = true;  // transpose B
constexpr uint32_t kInnerKTiles = 1;  // K tiles per matmul_block call

// M_block x N_block matmul with K-outer layout.
void matmul_blocks(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t M_block,
    uint32_t N_block,
    uint32_t K_block_tiles,
    uint32_t subblock_h,
    uint32_t subblock_w,
    uint32_t current_M,
    uint32_t current_N) {
    uint32_t last_sh = subblock_h, last_sw = subblock_w;

    for (uint32_t ms = 0; ms < current_M; ms += subblock_h) {
        uint32_t current_sh = std::min(subblock_h, current_M - ms);
        for (uint32_t ns = 0; ns < current_N; ns += subblock_w) {
            uint32_t current_sw = std::min(subblock_w, current_N - ns);

            // Only reconfigure when subblock size changes (edge tiles)
            if (current_sh != last_sh || current_sw != last_sw) {
                mm_block_init_short(in0_cb, in1_cb, kTransposeIn1, current_sw, current_sh, kInnerKTiles);
                last_sh = current_sh;
                last_sw = current_sw;
            }

            tile_regs_acquire();

            for (uint32_t k = 0; k < K_block_tiles; k++) {
                uint32_t in0_index = k * M_block + ms;
                uint32_t in1_index = k * N_block + ns;
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    kDstAccIdx,
                    kTransposeIn1,
                    current_sw,
                    current_sh,
                    kInnerKTiles);
            }

            tile_regs_commit();
            tile_regs_wait();

            uint32_t dst = 0;
            for (uint32_t h = 0; h < current_sh; h++) {
                for (uint32_t w = 0; w < current_sw; w++) {
                    uint32_t out_tile_id = (ms + h) * current_N + (ns + w);
                    pack_tile<true>(dst, out_cb, out_tile_id);
                    dst++;
                }
            }

            tile_regs_release();
        }
    }
    // Restore full subblock size for next call
    if (last_sh != subblock_h || last_sw != subblock_w) {
        mm_block_init_short(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
    }
}

// Pack c_2 → c_5 in row-major order for DM to send to reduction partner.
// REDUCE_SENDER_TRANSPOSE: transpose tile content + swap indices so receiver can add directly.
// REDUCE_SENDER: identity copy with FP32 → BF16 format conversion.
void pack_subblock_pernsb(uint32_t in_cb, uint32_t out_cb, uint32_t current_M, uint32_t current_N) {
#ifdef REDUCE_SENDER_TRANSPOSE
    transpose_wh_init(in_cb, out_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t m = 0; m < current_M; m++) {
        for (uint32_t n = 0; n < current_N; n++) {
            uint32_t in_tile = m * current_N + n;
            uint32_t out_tile = n * current_M + m;
            acquire_dst();
            transpose_wh_tile(in_cb, in_tile, 0);
            pack_tile<true>(0, out_cb, out_tile);
            release_dst();
        }
    }
#else
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t t = 0; t < current_M * current_N; t++) {
        acquire_dst();
        copy_tile(in_cb, t, 0);
        pack_tile<true>(0, out_cb, t);
        release_dst();
    }
#endif
}

#ifdef REDUCE_ACCUMULATOR
void add_reduce_block(uint32_t own_cb, uint32_t recv_cb, uint32_t out_cb, uint32_t M_rows, uint32_t N_cols) {
    add_tiles_init(own_cb, recv_cb);
    reconfig_data_format(own_cb, recv_cb);
    pack_reconfig_data_format(out_cb);

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_rows; m++) {
        cb_reserve_back(out_cb, N_cols);
        for (uint32_t n = 0; n < N_cols; n++) {
            acquire_dst();
            add_tiles(own_cb, recv_cb, tile_id, tile_id, 0);
            pack_tile(0, out_cb);
            release_dst();
            tile_id++;
        }
        cb_push_back(out_cb, N_cols);
    }
}

#ifdef MIRROR_OUTPUT
// Add own_cb + recv_cb, transpose via c_7 staging, pack to mirror_cb (col by col).
// Produces N_cols columns of M_rows tiles each (matching DM mirror write pattern).
// src_stride: column stride in source CBs (= M_block for row-major c_2).
// Note: transpose_wh_dest() is buggy on Blackhole (PCC≈0.2), so we stage through c_7 BF16 CB.
void add_transpose_block(
    uint32_t own_cb, uint32_t recv_cb, uint32_t mirror_cb, uint32_t M_rows, uint32_t N_cols, uint32_t src_stride) {
    constexpr uint32_t staging_cb = tt::CBIndex::c_7;
    for (uint32_t n = 0; n < N_cols; n++) {
        // Phase 1: batch add M_rows tiles for column n into staging
        add_tiles_init(own_cb, recv_cb);
        reconfig_data_format(own_cb, recv_cb);
        pack_reconfig_data_format(staging_cb);
        cb_reserve_back(staging_cb, M_rows);
        for (uint32_t m = 0; m < M_rows; m++) {
            uint32_t tile_id = n * src_stride + m;
            acquire_dst();
            add_tiles(own_cb, recv_cb, tile_id, tile_id, 0);
            pack_tile<true>(0, staging_cb, m);
            release_dst();
        }
        cb_push_back(staging_cb, M_rows);

        // Phase 2: transpose all M_rows tiles from staging to mirror
        cb_wait_front(staging_cb, M_rows);
        cb_reserve_back(mirror_cb, M_rows);
        transpose_wh_init(staging_cb, mirror_cb);
        reconfig_data_format_srca(staging_cb);
        pack_reconfig_data_format(mirror_cb);
        for (uint32_t m = 0; m < M_rows; m++) {
            acquire_dst();
            transpose_wh_tile(staging_cb, m, 0);
            pack_tile(0, mirror_cb);
            release_dst();
        }
        cb_pop_front(staging_cb, M_rows);
        cb_push_back(mirror_cb, M_rows);
    }
}
#endif
#endif

void kernel_main() {
    constexpr uint32_t K_half = get_compile_time_arg_val(0);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Mpc = get_compile_time_arg_val(2);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(3);
    constexpr uint32_t M_block = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t N_block = get_compile_time_arg_val(6);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t K_num_blocks = K_half / K_block_tiles;
    constexpr uint32_t tiles_per_in0_block = K_block_tiles * M_block;
    constexpr uint32_t tiles_per_in1_block = K_block_tiles * N_block;
    constexpr uint32_t num_m_blocks = (Mpc + M_block - 1) / M_block;

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t intermed_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_5;

#ifdef REDUCE_ACCUMULATOR
    constexpr uint32_t reduce_cb = tt::CBIndex::c_5;
    constexpr uint32_t combined_cb = tt::CBIndex::c_6;
#endif

    mm_init(in0_cb, in1_cb, intermed_cb);
    mm_block_init_short(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
    reconfig_data_format(in1_cb, in0_cb);
    pack_reconfig_data_format(intermed_cb);

    for (uint32_t m_sub = 0; m_sub < num_m_blocks; m_sub++) {
        uint32_t current_M_block = std::min(M_block, Mpc - m_sub * M_block);

        for (uint32_t n_sub = 0; n_sub < num_n_blocks; n_sub++) {
            uint32_t N_start = n_sub * N_block;
            uint32_t current_N = std::min(N_block, Mpc - N_start);
            uint32_t intermed_tiles = current_M_block * current_N;

            cb_reserve_back(intermed_cb, intermed_tiles);

            for (uint32_t kb = 0; kb < K_num_blocks; kb++) {
                cb_wait_front(in0_cb, tiles_per_in0_block);
                cb_wait_front(in1_cb, tiles_per_in1_block);

                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    intermed_cb,
                    M_block,
                    N_block,
                    K_block_tiles,
                    subblock_h,
                    subblock_w,
                    current_M_block,
                    current_N);

                cb_pop_front(in0_cb, tiles_per_in0_block);
                cb_pop_front(in1_cb, tiles_per_in1_block);

                if (kb == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermed_cb, intermed_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            // Pack or reduce immediately after each (m_sub, n_sub) block
#ifndef REDUCE_ACCUMULATOR
            // REDUCE_SENDER path: pack c_2 → c_5 for DM to send to partner
            cb_reserve_back(out_cb, intermed_tiles);
            cb_wait_front(intermed_cb, intermed_tiles);
            pack_subblock_pernsb(intermed_cb, out_cb, current_M_block, current_N);
            cb_pop_front(intermed_cb, intermed_tiles);
            cb_push_back(out_cb, intermed_tiles);
#else
            // REDUCE_ACCUMULATOR: add c_2(FP32) + c_5(BF16) → c_6
            cb_wait_front(intermed_cb, intermed_tiles);
            cb_wait_front(reduce_cb, intermed_tiles);
            add_reduce_block(intermed_cb, reduce_cb, combined_cb, current_M_block, current_N);
#ifdef MIRROR_OUTPUT
            add_transpose_block(intermed_cb, reduce_cb, tt::CBIndex::c_4, current_M_block, current_N, current_M_block);
#endif
            cb_pop_front(intermed_cb, intermed_tiles);
            cb_pop_front(reduce_cb, intermed_tiles);
#endif

            // Re-init matmul pipeline after copy/pack changed data formats
            if (n_sub + 1 < num_n_blocks) {
                mm_init(in0_cb, in1_cb, intermed_cb);
                mm_block_init_short(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
                reconfig_data_format(in1_cb, in0_cb);
                pack_reconfig_data_format(intermed_cb);
            }
        }

        // Re-init matmul pipeline for next m_sub
        if (m_sub + 1 < num_m_blocks) {
            mm_init(in0_cb, in1_cb, intermed_cb);
            mm_block_init_short(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermed_cb);
        }
    }
}
