// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"

#ifdef FUSE_SILU
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
#include "llk_math_eltwise_binary_sfpu_binop.h"
#endif
#endif

/**
 * DRAM streaming matmul compute kernel with subblocking.
 *
 * Each core computes: output[M=1, N=per_core_N] = in0[M=1, K] @ in1[K, N=per_core_N]
 *
 * in0 is fully available in L1 (replicated, tensor-backed CB).
 * in1 is streamed from DRAM, subblock_k tiles at a time.
 *
 * Two execution paths:
 * - FUSE_SILU: Per-tile pipelining with SFPU overlap (SiLU on PACK thread)
 * - No SILU: Batch processing of subblock_w tiles
 */

#ifdef FUSE_SILU
/**
 * Helper: Run SiLU on PACK thread (TRISC2) with SFPU overlap.
 * Waits for Math, flips DST offset, runs SiLU, stalls until SFPU done.
 */
template <uint32_t tile_r_dim>
FORCE_INLINE void run_silu_on_pack() {
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

    if constexpr (tile_r_dim <= 4) {
        PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 2>(0, (int)VectorMode::R)));
    } else if constexpr (tile_r_dim == 8) {
        PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 4>(0, (int)VectorMode::R)));
    } else {
        PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 8>(0, (int)VectorMode::R)));
    }

    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

/**
 * Helper: Process tiles with fused SiLU activation using SFPU overlap.
 * Per-tile pipelining: matmul -> commit -> SFPU -> pack for each tile.
 */
template <uint32_t tile_r_dim>
FORCE_INLINE void process_subblock_with_silu(
    uint32_t cb_id_in0,
    uint32_t cb_id_in1,
    uint32_t cb_id_out,
    uint32_t subblock_k,
    uint32_t subblock_w,
    uint32_t num_subblocks_k) {
    for (uint32_t w = 0; w < subblock_w; w++) {
        tile_regs_acquire();

        // Matmul: accumulate K subblocks into dest[0]
        // Intermediate subblocks: finalize=false (partial accumulation)
        for (uint32_t sb_k = 0; sb_k < num_subblocks_k - 1; sb_k++) {
            cb_wait_front(cb_id_in1, subblock_k);
            custom_mm_block<false>(cb_id_in0, cb_id_in1, sb_k * subblock_k, 0, 0, subblock_k);
            cb_pop_front(cb_id_in1, subblock_k);
        }
        // Final subblock: finalize=true
        cb_wait_front(cb_id_in1, subblock_k);
        custom_mm_block<true>(cb_id_in0, cb_id_in1, (num_subblocks_k - 1) * subblock_k, 0, 0, subblock_k);
        cb_pop_front(cb_id_in1, subblock_k);

        tile_regs_commit();
        run_silu_on_pack<tile_r_dim>();
        pack_tile(0, cb_id_out, w);
        tile_regs_release();
    }
}
#else
/**
 * Helper: Process tiles without SiLU (batch processing).
 * All tiles computed first, then packed together.
 */
FORCE_INLINE void process_subblock_no_silu(
    uint32_t cb_id_in0,
    uint32_t cb_id_in1,
    uint32_t cb_id_out,
    uint32_t subblock_k,
    uint32_t subblock_w,
    uint32_t num_subblocks_k) {
    tile_regs_acquire();

    // Compute all tiles in subblock
    for (uint32_t w = 0; w < subblock_w; w++) {
        // Intermediate subblocks: finalize=false (partial accumulation)
        for (uint32_t sb_k = 0; sb_k < num_subblocks_k - 1; sb_k++) {
            cb_wait_front(cb_id_in1, subblock_k);
            custom_mm_block<false>(cb_id_in0, cb_id_in1, sb_k * subblock_k, 0, w, subblock_k);
            cb_pop_front(cb_id_in1, subblock_k);
        }
        // Final subblock: finalize=true
        cb_wait_front(cb_id_in1, subblock_k);
        custom_mm_block<true>(cb_id_in0, cb_id_in1, (num_subblocks_k - 1) * subblock_k, 0, w, subblock_k);
        cb_pop_front(cb_id_in1, subblock_k);
    }

    tile_regs_commit();
    tile_regs_wait();

    // Pack all tiles
    for (uint32_t w = 0; w < subblock_w; w++) {
        pack_tile(w, cb_id_out, w);
    }
    tile_regs_release();
}
#endif

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(2);
    constexpr uint32_t subblock_k = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(6);
    constexpr uint32_t tile_r_dim = get_compile_time_arg_val(7);

    constexpr uint32_t num_subblocks_n = per_core_N / subblock_w;
    constexpr uint32_t num_tiles_k = subblock_k * num_subblocks_k;

#ifdef FUSE_SILU
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
#endif

    // Initialize custom matmul
    custom_mm_block_init<false, true>(cb_id_in0, cb_id_in1, cb_id_out);

    // Wait for all in0 tiles (replicated, tensor-backed - always available)
    cb_wait_front(cb_id_in0, num_tiles_k);

    for (uint32_t sb_n = 0; sb_n < num_subblocks_n; sb_n++) {
        cb_reserve_back(cb_id_out, subblock_w);

#ifdef FUSE_SILU
        process_subblock_with_silu<tile_r_dim>(
            cb_id_in0, cb_id_in1, cb_id_out, subblock_k, subblock_w, num_subblocks_k);
#else
        process_subblock_no_silu(cb_id_in0, cb_id_in1, cb_id_out, subblock_k, subblock_w, num_subblocks_k);
#endif

        cb_push_back(cb_id_out, subblock_w);
    }

    cb_pop_front(cb_id_in0, num_tiles_k);
}
