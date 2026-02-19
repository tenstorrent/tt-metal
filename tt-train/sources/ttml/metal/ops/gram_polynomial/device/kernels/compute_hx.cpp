// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Phase 3 compute kernel for newton_schulz_iteration: X' = H @ X + a*X.
// Adapted from Phase 2's compute_gsq.cpp.
// Key differences:
//   - Epilogue: a*X + HX instead of b*G + c*G^2
//   - Pass 2 has no scalar multiply — HX is used directly
//   - Scalar a passed as bit_cast<uint32_t>(float) compile-time arg
//

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

// Matmul block computation — same as Phase 2 (transpose=false)
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
                    false /*transpose: H*X, no transpose needed*/,
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
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t a_bits = get_compile_time_arg_val(8);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;
    constexpr uint32_t x_input_cb = tt::CBIndex::c_4;

    mm_init(in0_cb, in1_cb, intermediate_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    bool reuse_in0_block = false;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        current_M_block_tiles = m_tile_end - m_tile;
        current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            current_N_block_tiles = n_tile_end - n_tile;
            current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            mm_block_init_short(
                in0_cb,
                in1_cb,
                false /*transpose: H*X, no transpose needed*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);

            // Accumulation buffer for H@X
            cb_reserve_back(intermediate_cb, out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                cb_wait_front(in0_cb, in0_block_num_tiles);
                cb_wait_front(in1_cb, in1_block_num_tiles);

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

                if (k_block == K_num_blocks - 1) {
                    if (n_block_iter < N_blocks_per_core - 1) {
                        reuse_in0_block = true;
                    }
                }
                if (!reuse_in0_block) {
                    cb_pop_front(in0_cb, in0_block_num_tiles);
                }
                cb_pop_front(in1_cb, in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_push_back(intermediate_cb, out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            // ---- Fused epilogue: X'[m,n] = a * X[m,n] + H@X[m,n] ----
            //
            // Uses L1 accumulation (same pattern as Phase 2):
            //   Pass 1: pack a*X into out_cb (no L1 acc, base layer)
            //   Pass 2: add HX into out_cb (L1 acc ON, adds to existing)
            //
            // Key difference from Phase 2: Pass 2 has no scalar multiply —
            // HX is used directly. Only aX needs scaling.
            //
            cb_wait_front(intermediate_cb, out_block_num_tiles);
            cb_wait_front(x_input_cb, out_block_num_tiles);
            cb_reserve_back(out_cb, out_block_num_tiles);

            // Pass 1: a*X -> out_cb
            copy_tile_to_dst_init_short(x_input_cb);
            reconfig_data_format_srca(x_input_cb);
            binop_with_scalar_tile_init();
            pack_reconfig_data_format(out_cb);

            {
                uint32_t tile_id = 0;
                for (uint32_t m = 0; m < M_block_tiles; m++) {
                    for (uint32_t n = 0; n < N_block_tiles; n++) {
                        acquire_dst();
                        copy_tile(x_input_cb, tile_id, 0);
                        mul_unary_tile(0, a_bits);
                        pack_tile<true>(0, out_cb, tile_id);
                        release_dst();
                        tile_id++;
                    }
                }
            }

            // Pass 2: HX -> add to out_cb via L1 accumulation (no scalar multiply)
            PACK((llk_pack_reconfig_l1_acc(1)));
            copy_tile_to_dst_init_short_with_dt(x_input_cb, intermediate_cb);

            {
                uint32_t tile_id = 0;
                for (uint32_t m = 0; m < M_block_tiles; m++) {
                    for (uint32_t n = 0; n < N_block_tiles; n++) {
                        acquire_dst();
                        copy_tile(intermediate_cb, tile_id, 0);
                        pack_tile<true>(0, out_cb, tile_id);
                        release_dst();
                        tile_id++;
                    }
                }
            }

            PACK((llk_pack_reconfig_l1_acc(0)));

            // Push entire block at once (DM's granular writer handles row-by-row pop)
            cb_push_back(out_cb, out_block_num_tiles);

            cb_pop_front(intermediate_cb, out_block_num_tiles);
            cb_pop_front(x_input_cb, out_block_num_tiles);
        }
    }
}
