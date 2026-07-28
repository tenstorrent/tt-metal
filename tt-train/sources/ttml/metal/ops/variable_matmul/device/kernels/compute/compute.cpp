// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

// Renamed from copy_block to avoid an ambiguous overload with ckernel::copy_block (added to
// api/compute/tile_move_copy.h in #49070), which has the identical (uint32_t, uint32_t, uint32_t,
// uint32_t) signature and is in scope here. See tt-metal#50386.
void copy_and_pack_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    constexpr uint32_t dst_id = 0;
    CircularBuffer out_buf(out_cb);

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            copy_tile(in_cb, tile_id, dst_id /*dst*/);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_id, out_cb);
            tile_regs_release();
            tile_id++;
        }
        out_buf.push_back(N_block_tiles);
    }
}

// Transpose every tile of a block from src_cb into dst_cb.
template <uint32_t NumTiles, uint32_t ChunkSize = 4>
inline void transpose_in0_block_streamed(uint32_t src_cb, uint32_t dst_cb) {
    constexpr uint32_t kFullChunks = NumTiles / ChunkSize;
    constexpr uint32_t kTail = NumTiles % ChunkSize;

    PACK((llk_pack_reconfig_l1_acc(0)));

    CircularBuffer src_buf(src_cb);
    CircularBuffer dst_buf(dst_cb);
    for (uint32_t b = 0; b < kFullChunks; ++b) {
        src_buf.wait_front(ChunkSize);
        tile_regs_acquire();
        for (uint32_t j = 0; j < ChunkSize; ++j) {
            transpose_wh_tile(src_cb, j, /*dst=*/j);
        }
        tile_regs_commit();
        src_buf.pop_front(ChunkSize);

        dst_buf.reserve_back(ChunkSize);
        tile_regs_wait();
        for (uint32_t j = 0; j < ChunkSize; ++j) {
            pack_tile(j, dst_cb);
        }
        tile_regs_release();
        dst_buf.push_back(ChunkSize);
    }
    if constexpr (kTail > 0) {
        src_buf.wait_front(kTail);
        tile_regs_acquire();
        for (uint32_t j = 0; j < kTail; ++j) {
            transpose_wh_tile(src_cb, j, /*dst=*/j);
        }
        tile_regs_commit();
        src_buf.pop_front(kTail);

        dst_buf.reserve_back(kTail);
        tile_regs_wait();
        for (uint32_t j = 0; j < kTail; ++j) {
            pack_tile(j, dst_cb);
        }
        tile_regs_release();
        dst_buf.push_back(kTail);
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
    const uint32_t subblock_w,
    const uint32_t transpose_b) {
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
                    transpose_b /*transpose*/,
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
                const uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    const uint32_t w_tile_id = N_start + w;
                    const uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
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

// Write zero tiles to `out_cb` for one M×N block. K=0 path (empty-expert dW): no matmul
// runs, so without this the pack would copy uninitialized DST into the weight gradient.
void zero_blocks(
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    fill_tile_init();
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();
            uint32_t dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w++) {
                    fill_tile(dst_index, 0.0F);
                    dst_index++;
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                const uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    const uint32_t w_tile_id = N_start + w;
                    const uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                }
            }
            tile_regs_release();
        }
    }
}

void kernel_main() {
    // K_tiles, M_blocks_per_core, and the M/N tile ranges come from runtime args (read below);
    // K_num_blocks is derived from K_tiles further down.
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t transpose_b = get_compile_time_arg_val(6);
    constexpr bool transpose_a = static_cast<bool>(get_compile_time_arg_val(7));

    uint32_t argidx = 0;
    // OFFSET_ROW_MODE overrides M_start/M_end/M_blocks_per_core via cb_ctrl below.
    uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t M_blocks_per_core = get_arg_val<uint32_t>(argidx++);
    // OFFSET_K_MODE overrides K_tiles from cb_ctrl[3].
    uint32_t K_tiles = get_arg_val<uint32_t>(argidx++);

    // cb_ctrl payload, packed by whichever dm kernel owns the publish (see dataflow kernels):
    //   OFFSET_ROW_MODE: ctrl[0..2] = (M_start, M_end, M_blocks_per_core)
    //   OFFSET_K_MODE:   ctrl[3]    = K_tiles
    // The two modes are mutually exclusive (one per role), so exactly one payload is written
    // per invocation.
    {
        CircularBuffer cb_ctrl(tt::CBIndex::c_8);
        cb_ctrl.wait_front(1U);
#ifdef OFFSET_ROW_MODE
        M_start_tile = cb_ctrl.read_tile_value(0U, 0U);
        M_end_tile = cb_ctrl.read_tile_value(0U, 1U);
        M_blocks_per_core = cb_ctrl.read_tile_value(0U, 2U);
#else
        K_tiles = cb_ctrl.read_tile_value(0U, 3U);
#endif
        cb_ctrl.pop_front(1U);
    }
    const uint32_t padded_K_tiles = ((K_tiles + K_block_tiles - 1U) / K_block_tiles) * K_block_tiles;
    const uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    // When transpose_a is set, the dataflow writes the raw [K, M]-stored block into in0_cb,
    // and the compute kernel transposes each tile into in0_transposed_cb (c_7), which is
    // what the matmul actually consumes. `in0_cb_for_matmul` selects the right one.
    constexpr uint32_t in0_transposed_cb = tt::CBIndex::c_7;
    constexpr uint32_t in0_cb_for_matmul = transpose_a ? in0_transposed_cb : in0_cb;

    CircularBuffer cb_in0_mm(in0_cb_for_matmul);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_interm(intermediate_cb);

    mm_init(in0_cb_for_matmul, in1_cb, intermediate_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    bool reuse_in0_block = false;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        current_M_block_tiles = m_tile_end - m_tile;
        current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            const uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            const uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            current_N_block_tiles = n_tile_end - n_tile;
            current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            mm_block_init_short(
                in0_cb_for_matmul,
                in1_cb,
                transpose_b /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            reconfig_data_format(in1_cb, in0_cb_for_matmul);
            pack_reconfig_data_format(intermediate_cb);
            // Disable L1 packer accumulator before k=0 pack so matmul packs cleanly over
            // intermediate_cb instead of adding onto any leftover state from a prior program.
            PACK((llk_pack_reconfig_l1_acc(0)));
            cb_interm.reserve_back(out_block_num_tiles);
            if (K_num_blocks == 0U) {
                // Empty K-axis offset (empty expert): K-loop skipped, intermediate_cb would
                // hold uninitialized state. Zero the FULL M_block × N_block region
                // (copy_and_pack_block later reads it whole) so `add_grad` downstream contributes nothing.
                zero_blocks(intermediate_cb, M_block_tiles, N_block_tiles, N_block_tiles, subblock_h, subblock_w);
            }
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                if constexpr (transpose_a) {
                    // Mirror the dataflow's skip-push pattern: dataflow skips pushing in0 at
                    // k=0 of any non-first N-iter (the slice is identical to the previous
                    // N-iter's). Skip the transpose pass in exactly that case so c_0 reads
                    // and c_7 pushes stay in sync.
                    // NB: reuse_in0_block is reset at end of every iter, so at the *start* of
                    // this reuse iter it is already false — we cannot rely on it here.
                    const bool is_reuse_iter = (k_block == 0) && (n_block_iter > 0);
                    if (!is_reuse_iter) {
                        DeviceZoneScopedN("TRANSPOSE-A");
                        // State setup for transpose (one-time per K-iter, before streaming).
                        reconfig_data_format_srca(in1_cb, in0_cb);
                        transpose_wh_init_short(in0_cb);
                        pack_reconfig_data_format(in0_transposed_cb);
                        transpose_in0_block_streamed<in0_block_num_tiles>(in0_cb, in0_transposed_cb);
                        // Restore matmul state. The "_with_dt" variant handles the srcA data
                        // format switch from in0_cb to in0_cb_for_matmul in one go.
                        mm_block_init_short_with_dt(
                            in0_cb_for_matmul,
                            in1_cb,
                            in0_cb,
                            transpose_b,
                            current_subblock_w,
                            current_subblock_h,
                            K_block_tiles);
                        pack_reconfig_data_format(intermediate_cb);
                        // transpose_in0_block_streamed disabled the L1 packer accumulator so
                        // its packs would overwrite. Restore it to the right state for the
                        // matmul pack: enabled after k=0 (so the matmul accumulates into
                        // intermediate_cb across K-iterations), disabled at k=0 itself.
                        PACK((llk_pack_reconfig_l1_acc(k_block == 0 ? 0 : 1)));
                    }
                }

                {
                    DeviceZoneScopedN("MATMUL-K-ITER");
                    cb_in0_mm.wait_front(in0_block_num_tiles);
                    cb_in1.wait_front(in1_block_num_tiles);

                    matmul_blocks(
                        in0_cb_for_matmul,
                        in1_cb,
                        intermediate_cb,
                        current_M_block_tiles,
                        current_N_block_tiles,
                        N_block_tiles,
                        K_block_tiles,
                        current_subblock_h,
                        current_subblock_w,
                        transpose_b);
                }

                if (k_block == K_num_blocks - 1) {
                    // Last K-iter: if the next N-block reuses this M-block's in0, keep it (the
                    // !reuse_in0_block check below then skips popping in0_cb).
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
                    }
                }
                if (!reuse_in0_block) {
                    cb_in0_mm.pop_front(in0_block_num_tiles);
                }
                cb_in1.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_interm.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            cb_out.reserve_back(out_block_num_tiles);
            cb_interm.wait_front(out_block_num_tiles);
            copy_and_pack_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_interm.pop_front(out_block_num_tiles);
        }
    }
}
