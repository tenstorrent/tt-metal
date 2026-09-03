// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Matmul compute kernel. Computes M_block × N_block output blocks for
// the Mpc × Mpc gram matmul. See k_split_gram_matmul.hpp for the algorithm overview.
//
// Iteration structure: outer (m_sub, n_sub) loop over the Mpc × Mpc output, inner K-loop
// accumulates K_block_tiles tiles per pass into c_2 (FP32 intermediate). After each
// (m_sub, n_sub) block, the output is packed/sent/reduced per the core's role:
//
//   REDUCE_SENDER_TRANSPOSE:
//     matmul → c_2, then pack c_2 → c_5 transposed (tile content transposed + indices
//     swapped) so the receiver can add directly. DM writer NOC-sends c_5 to the
//     reduction partner. Diagonal cores use this path too: their even-K partial is
//     symmetric ((i,j) and (j,i) accumulate the same products in the same order), so the
//     transposed block matches the direct one — bitwise when multiplies are exact (HiFi4,
//     the default; lower fidelities pair the src operands asymmetrically, leaving
//     ulp-level differences).
//   REDUCE_ACCUMULATOR:
//     matmul → c_2, then add own c_2 (FP32) + partner's c_5 (BF16) → c_6. DM writer
//     writes c_6 to DRAM. If MIRROR_OUTPUT is defined, also stage-add+transpose into c_4
//     for the lower-triangle mirror tile (see add_transpose_block).
//
// Sub-block iteration order depends on the role. REDUCE_ACCUMULATOR cores iterate
// (m_sub, n_sub) row-major. REDUCE_SENDER_TRANSPOSE cores iterate column-major
// (n_sub outer, m_sub inner): the sender's transposed sub-block (m, n) is the
// receiver's sub-block (n, m), so column-major production delivers blocks in exactly
// the receiver's row-major consume order — one block in flight, no reordering buffer.
// The mcast senders feed each parity group in the matching order.
//
// CB discipline: every c_2/c_5 transaction is a full M_block × N_block block and every
// c_4/c_6/c_7 transaction is a full M_block group, with partial edge blocks occupying
// a valid prefix. CB pointers only wrap when they hit the buffer limit exactly, so
// pushes must always equal the CB capacity; the fixed sizes also pin c_5 to its base
// address on both ends of the reduce NOC write.
//
// Helper functions:
//   matmul_blocks        — K-loop matmul over a subblock grid.
//   pack_transposed_block — c_2 → c_5 (REDUCE_SENDER_TRANSPOSE).
//   add_reduce_block     — c_2 + c_5 → c_6 (REDUCE_ACCUMULATOR).
//   add_transpose_block  — c_2 + c_5 → c_4 via c_7 staging (MIRROR_OUTPUT).

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"

constexpr uint32_t kDstAccIdx = 0;    // DST tile index for accumulation
constexpr bool kTransposeIn1 = true;  // transpose B
constexpr uint32_t kInnerKTiles = 1;  // K tiles per matmul_block call

// M_block x N_block matmul with K-outer layout.
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block,
    const uint32_t N_block,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const uint32_t current_M,
    const uint32_t current_N) {
    uint32_t last_sh = subblock_h, last_sw = subblock_w;

    for (uint32_t ms = 0; ms < current_M; ms += subblock_h) {
        const uint32_t current_sh = std::min(subblock_h, current_M - ms);
        for (uint32_t ns = 0; ns < current_N; ns += subblock_w) {
            const uint32_t current_sw = std::min(subblock_w, current_N - ns);

            // Only reconfigure when subblock size changes (edge tiles)
            if (current_sh != last_sh || current_sw != last_sw) {
                matmul_block_init(in0_cb, in1_cb, kTransposeIn1, current_sw, current_sh, kInnerKTiles);
                last_sh = current_sh;
                last_sw = current_sw;
            }

            tile_regs_acquire();

            for (uint32_t k = 0; k < K_block_tiles; k++) {
                const uint32_t in0_index = k * M_block + ms;
                const uint32_t in1_index = k * N_block + ns;
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
                    const uint32_t out_tile_id = (ms + h) * current_N + (ns + w);
                    pack_tile<true>(dst, out_cb, out_tile_id);
                    dst++;
                }
            }

            tile_regs_release();
        }
    }
    // Restore full subblock size for next call
    if (last_sh != subblock_h || last_sw != subblock_w) {
        matmul_block_init(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
    }
}

// Pack in_cb → out_cb transposed (tile content transposed + indices swapped) so the
// reduction partner can add directly. The packed block is the partner's sub-block,
// row-major: current_N rows × current_M cols, compact at the front of the reservation.
void pack_transposed_block(
    const uint32_t in_cb, const uint32_t out_cb, const uint32_t current_M, const uint32_t current_N) {
    transpose_init(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t m = 0; m < current_M; m++) {
        for (uint32_t n = 0; n < current_N; n++) {
            const uint32_t in_tile = m * current_N + n;
            const uint32_t out_tile = n * current_M + m;
            tile_regs_acquire();
            transpose_tile(in_cb, in_tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb, out_tile);
            tile_regs_release();
        }
    }
}

#ifdef REDUCE_ACCUMULATOR
// row_group: fixed CB transaction size (= c_6 capacity); valid tiles are the prefix.
void add_reduce_block(
    const uint32_t own_cb,
    const uint32_t recv_cb,
    const uint32_t out_cb,
    const uint32_t M_rows,
    const uint32_t N_cols,
    const uint32_t row_group) {
    add_init(own_cb, recv_cb);
    reconfig_data_format(own_cb, recv_cb);
    pack_reconfig_data_format(out_cb);

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_rows; m++) {
        cb_reserve_back(out_cb, row_group);
        for (uint32_t n = 0; n < N_cols; n++) {
            tile_regs_acquire();
            add_tiles(own_cb, recv_cb, tile_id, tile_id, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();
            tile_id++;
        }
        cb_push_back(out_cb, row_group);
    }
}

#ifdef MIRROR_OUTPUT
// Add own_cb + recv_cb, transpose via c_7 staging, pack to mirror_cb.
// Group n (one per source column, N_cols total) carries mirror row n of the block:
// source tiles (m, n) for m = 0..M_rows, each tile-transposed. src_stride is the
// source row stride (= current_N for the compact row-major block). Every staging and
// mirror CB transaction is `group_capacity` tiles (= CB capacity) so pushes always
// wrap exactly; valid tiles are the prefix.
// Note: transpose_dest() is buggy on Blackhole (PCC≈0.2), so we stage through c_7 BF16 CB.
void add_transpose_block(
    const uint32_t own_cb,
    const uint32_t recv_cb,
    const uint32_t mirror_cb,
    const uint32_t M_rows,
    const uint32_t N_cols,
    const uint32_t src_stride,
    const uint32_t group_capacity) {
    constexpr uint32_t staging_cb = tt::CBIndex::c_7;
    for (uint32_t n = 0; n < N_cols; n++) {
        // Phase 1: batch add M_rows tiles of source column n into staging
        add_init(own_cb, recv_cb);
        reconfig_data_format(own_cb, recv_cb);
        pack_reconfig_data_format(staging_cb);
        cb_reserve_back(staging_cb, group_capacity);
        for (uint32_t m = 0; m < M_rows; m++) {
            const uint32_t tile_id = m * src_stride + n;
            tile_regs_acquire();
            add_tiles(own_cb, recv_cb, tile_id, tile_id, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, staging_cb, m);
            tile_regs_release();
        }
        cb_push_back(staging_cb, group_capacity);

        // Phase 2: transpose all M_rows tiles from staging to mirror
        cb_wait_front(staging_cb, group_capacity);
        cb_reserve_back(mirror_cb, group_capacity);
        transpose_init(staging_cb);
        reconfig_data_format_srca(staging_cb);
        pack_reconfig_data_format(mirror_cb);
        for (uint32_t m = 0; m < M_rows; m++) {
            tile_regs_acquire();
            transpose_tile(staging_cb, m, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, mirror_cb);
            tile_regs_release();
        }
        cb_pop_front(staging_cb, group_capacity);
        cb_push_back(mirror_cb, group_capacity);
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
    // The sub-block loop remap below relies on the grids being square.
    static_assert(num_m_blocks == num_n_blocks);
    // Fixed transaction size for c_2/c_5 (= their capacity); partial blocks use a prefix.
    constexpr uint32_t block_capacity = M_block * N_block;

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t intermed_cb = tt::CBIndex::c_2;

#ifndef REDUCE_ACCUMULATOR
    constexpr uint32_t out_cb = tt::CBIndex::c_5;
#else
    constexpr uint32_t reduce_cb = tt::CBIndex::c_5;
    constexpr uint32_t combined_cb = tt::CBIndex::c_6;
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermed_cb);
    matmul_block_init(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
    reconfig_data_format(in1_cb, in0_cb);
    pack_reconfig_data_format(intermed_cb);

    for (uint32_t outer = 0; outer < num_m_blocks; outer++) {
        for (uint32_t inner = 0; inner < num_n_blocks; inner++) {
            // Accumulators iterate row-major; senders column-major so transposed blocks
            // arrive in the partner's consume order (see header comment). The mcast feed
            // matches: even-parity senders stream m_sub fast, odd-parity m_sub slow.
#ifdef REDUCE_ACCUMULATOR
            const uint32_t m_sub = outer;
            const uint32_t n_sub = inner;
#else
            const uint32_t m_sub = inner;
            const uint32_t n_sub = outer;
#endif
            const uint32_t current_M_block = std::min(M_block, Mpc - m_sub * M_block);
            const uint32_t current_N = std::min(N_block, Mpc - n_sub * N_block);

            cb_reserve_back(intermed_cb, block_capacity);

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

            cb_push_back(intermed_cb, block_capacity);
            PACK((llk_pack_reconfig_l1_acc(0)));

            // Pack or reduce immediately after each (m_sub, n_sub) block
#ifndef REDUCE_ACCUMULATOR
            // Sender path: pack c_2 → c_5 transposed for DM to send to partner
            cb_reserve_back(out_cb, block_capacity);
            cb_wait_front(intermed_cb, block_capacity);
            pack_transposed_block(intermed_cb, out_cb, current_M_block, current_N);
            cb_pop_front(intermed_cb, block_capacity);
            cb_push_back(out_cb, block_capacity);
#else
            // REDUCE_ACCUMULATOR: add c_2(FP32) + c_5(BF16) → c_6
            cb_wait_front(intermed_cb, block_capacity);
            cb_wait_front(reduce_cb, block_capacity);
            add_reduce_block(intermed_cb, reduce_cb, combined_cb, current_M_block, current_N, N_block);
#ifdef MIRROR_OUTPUT
            add_transpose_block(
                intermed_cb, reduce_cb, tt::CBIndex::c_4, current_M_block, current_N, current_N, M_block);
#endif
            cb_pop_front(intermed_cb, block_capacity);
            cb_pop_front(reduce_cb, block_capacity);
#endif

            // Re-init matmul pipeline after copy/pack changed data formats
            if (inner + 1 < num_n_blocks) {
                matmul_block_init(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
                reconfig_data_format(in1_cb, in0_cb);
                pack_reconfig_data_format(intermed_cb);
            }
        }

        // Re-init matmul pipeline for next outer pass
        if (outer + 1 < num_m_blocks) {
            matmul_block_init(in0_cb, in1_cb, kTransposeIn1, subblock_w, subblock_h, kInnerKTiles);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermed_cb);
        }
    }
}
