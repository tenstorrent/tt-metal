// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute: Multi-expert two-phase expert compute kernel + FPU combine
//
// Phase 1 — Expert compute (for each expert e = 0..num_experts-1):
//   For each tile row m = 0..M_tiles-1:
//     Phase A: gate_up matmul [1, K_gu] x [K_gu, N_per_core_gu] -> SwiGLU -> CB_OUT
//     Phase B: down matmul [1, K_dn] x [K_dn, N_per_core_dn] -> CB_OUT
//
// Phase 2 — FPU combine (replaces scalar combine_dm_fused.cpp):
//   For each expert e = 0..num_experts-1:
//     For each tile row tr = 0..output_M_tiles-1:
//       Reader pushes output tile -> CB_COMBINE_OUT, expert tile -> CB_COMBINE_EXP
//       Compute: output_tile += weight * expert_tile
//       Writer: writes accumulated result back to output DRAM
//   Each compute core handles n_per_core_dn tile columns of the D dimension.
//
// CB0 (activation): 1 BF16 tile per K iteration
// CB1 (weights): max(N_per_core_gu, N_per_core_dn) BFP4 tiles per K iteration
// CB2 (output): max(N_per_core_gu/2, N_per_core_dn) BF16 tiles
// CB3 (combine_out): 1 BF16 tile — output accumulator for combine
// CB4 (combine_exp): 1 BF16 tile — expert out_buf tile for combine
// CB5 (combine_result): 1 BF16 tile — result of combine for writer

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"

#ifdef TRISC_PACK
#include "swiglu_sfpu.h"
#endif

void kernel_main() {
    constexpr uint32_t num_k_tiles_gu = get_compile_time_arg_val(0);
    constexpr uint32_t n_per_core_gu = get_compile_time_arg_val(1);
    constexpr uint32_t n_per_core_dn = get_compile_time_arg_val(2);
    constexpr uint32_t num_experts = get_compile_time_arg_val(3);
    constexpr uint32_t N_BLOCK_GU = get_compile_time_arg_val(4);
    constexpr uint32_t N_BLOCK_DN = get_compile_time_arg_val(5);
    constexpr uint32_t num_k_tiles_dn = get_compile_time_arg_val(6);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t output_M_tiles = get_compile_time_arg_val(8);  // output tile rows for combine
    constexpr uint32_t enable_combine = get_compile_time_arg_val(9);  // 1 to enable FPU combine
    constexpr uint32_t n_swiglu = n_per_core_gu / 2;

    constexpr uint32_t n_blocks_gu = n_per_core_gu / N_BLOCK_GU;
    constexpr uint32_t n_blocks_dn = n_per_core_dn / N_BLOCK_DN;

    constexpr uint32_t cb_act = 0;
    constexpr uint32_t cb_weights = 1;
    constexpr uint32_t cb_out = 2;
    constexpr uint32_t cb_combine_out = 3;  // output tile for combine
    constexpr uint32_t cb_combine_exp = 4;  // expert tile for combine
    constexpr uint32_t cb_combine_res = 5;  // result tile for combine

    // ========== Phase 1: Expert compute ==========
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        // Phase A: gate_up matmul + SwiGLU
        for (uint32_t m = 0; m < M_tiles; ++m) {
            mm_block_init(cb_act, cb_weights, cb_out, false, N_BLOCK_GU, 1, 1);

            tile_regs_acquire();

            for (uint32_t k = 0; k < num_k_tiles_gu; ++k) {
                cb_wait_front(cb_act, 1);
                cb_wait_front(cb_weights, n_per_core_gu);

                for (uint32_t b = 0; b < n_blocks_gu; ++b) {
                    uint32_t n_offset = b * N_BLOCK_GU;
                    matmul_block(cb_act, cb_weights, 0, n_offset, n_offset, false, N_BLOCK_GU, 1, 1);
                }

                cb_pop_front(cb_act, 1);
                cb_pop_front(cb_weights, n_per_core_gu);
            }

            tile_regs_commit();
            tile_regs_wait();

#ifdef TRISC_PACK
            ckernel::llk_math_eltwise_binary_sfpu_swiglu_init<true>();
            for (uint32_t j = 0; j < n_swiglu; ++j) {
                ckernel::llk_math_eltwise_binary_sfpu_swiglu<true, false>(j, j + n_swiglu, j);
            }
#endif

            cb_reserve_back(cb_out, n_swiglu);
            for (uint32_t i = 0; i < n_swiglu; ++i) {
                pack_tile(i, cb_out);
            }
            cb_push_back(cb_out, n_swiglu);

            tile_regs_release();
        }

        // Phase B: down matmul
        for (uint32_t m = 0; m < M_tiles; ++m) {
            mm_block_init(cb_act, cb_weights, cb_out, false, N_BLOCK_DN, 1, 1);

            tile_regs_acquire();

            for (uint32_t k = 0; k < num_k_tiles_dn; ++k) {
                cb_wait_front(cb_act, 1);
                cb_wait_front(cb_weights, n_per_core_dn);

                for (uint32_t b = 0; b < n_blocks_dn; ++b) {
                    uint32_t n_offset = b * N_BLOCK_DN;
                    matmul_block(cb_act, cb_weights, 0, n_offset, n_offset, false, N_BLOCK_DN, 1, 1);
                }

                cb_pop_front(cb_act, 1);
                cb_pop_front(cb_weights, n_per_core_dn);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, n_per_core_dn);
            for (uint32_t i = 0; i < n_per_core_dn; ++i) {
                pack_tile(i, cb_out);
            }
            cb_push_back(cb_out, n_per_core_dn);

            tile_regs_release();
        }
    }

    // ========== Phase 2: FPU combine ==========
    // For each expert, for each output tile row, for each tile column handled by this core:
    //   output_tile += weight * expert_tile
    // Reader feeds tiles one at a time via CB3 (output) and CB4 (expert).
    // We add them and pack to CB5 for writer.
    //
    // For K=1 weight=1.0, this is just add_tiles.
    // For general K>1, reader will also push a scalar weight tile to a CB
    // and we'd use mul_tiles_bcast_scalar + add_tiles. Deferred to later step.
    if constexpr (enable_combine) {
        // Initialize eltwise binary add (must be done before first add_tiles call)
        binary_op_init_common(cb_combine_out, cb_combine_exp, cb_combine_res);
        add_tiles_init(cb_combine_out, cb_combine_exp);

        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            for (uint32_t tr = 0; tr < output_M_tiles; ++tr) {
                for (uint32_t d = 0; d < n_per_core_dn; ++d) {
                    // Wait for reader to push output tile and expert tile
                    cb_wait_front(cb_combine_out, 1);
                    cb_wait_front(cb_combine_exp, 1);

                    // Add tiles: dst[0] = output_tile + expert_tile
                    tile_regs_acquire();
                    add_tiles(cb_combine_out, cb_combine_exp, 0, 0, 0);
                    tile_regs_commit();

                    cb_pop_front(cb_combine_out, 1);
                    cb_pop_front(cb_combine_exp, 1);

                    // Pack result to CB5 for writer
                    tile_regs_wait();
                    cb_reserve_back(cb_combine_res, 1);
                    pack_tile(0, cb_combine_res);
                    cb_push_back(cb_combine_res, 1);
                    tile_regs_release();
                }
            }
        }
    }
}
