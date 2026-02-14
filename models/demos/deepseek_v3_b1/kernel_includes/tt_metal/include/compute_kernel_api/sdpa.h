// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_sdpa_bcast_col_srcb_reuse_api.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_sdpa_bcast_col_srca_srcb_reuse_api.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_sdpa_reduce_row.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_sdpa_api.h"
#endif

namespace ckernel {

template <EltwiseBinaryType eltwise_binary_type = ELWADD, uint32_t num_tiles>
ALWI void sdpa_bcast_col_reuse_tiles_init(uint32_t icb0) {
    UNPACK((llk_unpack_A_sdpa_init<num_tiles, BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(
        false, false, icb0)));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse_init_with_operands<eltwise_binary_type, num_tiles, MATH_FIDELITY>(
        icb0, icb0, false)));
}

template <bool clear_dest = false>
ALWI void sdpa_bcast_col_reuse_preamble() {
    UNPACK((llk_unpack_A_sdpa_set_srcb_dummy_valid()));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse_preamble<DST_SYNC_MODE, DST_ACCUM_MODE, clear_dest>()));
}

ALWI void sdpa_bcast_col_reuse_postamble() { MATH((llk_math_sdpa_bcast_col_srcb_reuse_postamble())); }

template <EltwiseBinaryType eltwise_binary_type = ELWADD, uint32_t num_tiles>
ALWI void sdpa_bcast_col_reuse_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(in0_cb_id, in_tile_index)));
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(in1_cb_id, in_tile_index)));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse<eltwise_binary_type, num_tiles, DST_ACCUM_MODE, MATH_FIDELITY>(
        dst_tile_index)));
}

template <uint32_t num_tiles>
ALWI void sdpa_mul_bcast_col_reuse_tiles_init(uint32_t icb0) {
    sdpa_bcast_col_reuse_tiles_init<ELWMUL, num_tiles>(icb0);
}

template <uint32_t num_tiles>
ALWI void sdpa_mul_bcast_col_reuse_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    sdpa_bcast_col_reuse_tiles<ELWMUL, num_tiles>(in0_cb_id, in1_cb_id, in_tile_index, dst_tile_index);
}

template <EltwiseBinaryType eltwise_binary_type = ELWADD, uint32_t num_tiles>
ALWI void sdpa_bcast_col_srca_srcb_reuse_tiles_init(uint32_t icb0) {
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse_init_with_operands<eltwise_binary_type, num_tiles, MATH_FIDELITY>(
        icb0, icb0, false)));
}

template <bool clear_dest = false>
ALWI void sdpa_bcast_col_srca_srcb_reuse_preamble(uint32_t isrc) {
    UNPACK((llk_unpack_A_sdpa_set_srca_srcb_dummy_valid()));
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble<DST_SYNC_MODE, DST_ACCUM_MODE, clear_dest>(isrc)));
}

template <
    EltwiseBinaryType eltwise_binary_type = ELWADD,
    uint32_t num_tiles,
    bool skip_signalling = false,
    bool fused_signalling = false>
ALWI void sdpa_bcast_col_srca_srcb_reuse_tiles(uint32_t dst_tile_index) {
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse<
          eltwise_binary_type,
          num_tiles,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          skip_signalling,
          fused_signalling>(dst_tile_index)));
}

template <uint32_t num_tiles>
ALWI void sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init(uint32_t icb0) {
    sdpa_bcast_col_srca_srcb_reuse_tiles_init<ELWSUB, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool skip_signalling = false, bool fused_signalling = false>
ALWI void sdpa_sub_bcast_col_srca_srcb_reuse_tiles(uint32_t dst_tile_index) {
    sdpa_bcast_col_srca_srcb_reuse_tiles<ELWSUB, num_tiles, skip_signalling, fused_signalling>(dst_tile_index);
}

template <uint32_t num_tiles>
ALWI void sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init(uint32_t icb0) {
    sdpa_bcast_col_srca_srcb_reuse_tiles_init<ELWMUL, num_tiles>(icb0);
}

template <uint32_t num_tiles, bool skip_signalling = false, bool fused_signalling = false>
ALWI void sdpa_mul_bcast_col_srca_srcb_reuse_tiles(uint32_t dst_tile_index) {
    sdpa_bcast_col_srca_srcb_reuse_tiles<ELWMUL, num_tiles, skip_signalling, fused_signalling>(dst_tile_index);
}

template <DataFormat format>
ALWI void sdpa_reduce_row_init() {
    MATH((llk_math_sfpu_sdpa_reduce_row_init<APPROX, DST_ACCUM_MODE, format>()));
}

template <DataFormat format, uint32_t block_width>
ALWI void sdpa_reduce_max_row(uint src_index, uint dst_index, bool prev_max = false) {
    MATH((llk_math_sfpu_sdpa_reduce_max_row<APPROX, DST_ACCUM_MODE, format, block_width>(
        src_index, dst_index, prev_max)));
}

template <DataFormat format, uint32_t block_width>
ALWI void sdpa_reduce_sum_row(uint src_index, uint dst_index, bool prev_sum = false) {
    MATH((llk_math_sfpu_sdpa_reduce_sum_row<APPROX, DST_ACCUM_MODE, format, block_width>(
        src_index, dst_index, prev_sum)));
}

}  // namespace ckernel

// =============================================================================
// SDPA Tail Reduction - Fused SFPI Kernel and Helper
// =============================================================================

#ifdef TRISC_MATH

/**
 * The custom SFPI LLK function computes the following operation:
 * cur_max = max(prev_max, worker_max)
 * cur_sum = exp((worker_max - cur_max) * scale) * worker_sum + exp((prev_max - cur_max) * scale) * prev_sum
 * There are 4 results produced:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale), produced in dst_reg[worker_max_base_idx]
 * 3. cur_sum produced in dst_reg[prev_sum_base_idx]
 * 4. cur_max produced in dst_reg[cur_max_base_idx]
 * If final_norm is true, the output is:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale) * recip(cur_sum), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale) * recip(cur_sum), produced in dst_reg[worker_max_base_idx]
 * fused_max_sub_exp_add_tile
 */
template <bool SDPA_EXP_APPROX_MODE, bool final_norm = false>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    // Non-Approx mode for exp initializes recip for final normalization
    static_assert(!(final_norm && SDPA_EXP_APPROX_MODE), "Approx mode must be disabled when final_norm is true");

    // 8 rows
    constexpr int ITERATIONS_HALF_FACE = 2;
    constexpr uint32_t prev_max_base_idx = 0;     // Tile 0, col 0
    constexpr uint32_t prev_sum_base_idx = 1;     // Tile 0, col 1
    constexpr uint32_t worker_max_base_idx = 32;  // Tile 1, col 0
    constexpr uint32_t worker_sum_base_idx = 33;  // Tile 1, col 1
    constexpr uint32_t cur_max_base_idx = 64;     // Tile 2, col 0 (output)
    constexpr uint32_t cur_sum_base_idx = 65;     // Tile 2, col 1 (output)

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        // Load inputs for this vector-slot into temporaries to avoid aliasing on dst_reg
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat cur_max;
        v_if(prev_max_vec < worker_max_vec) { cur_max = worker_max_vec; }
        v_else { cur_max = prev_max_vec; }
        v_endif;
        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_max_base_idx] = cur_max;
        }

        // Compute differences
        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        // Exponentials of differences
        sfpi::vFloat exp_prev = ckernel::sfpu::
            _calculate_exponential_piecewise_<SDPA_EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker = ckernel::sfpu::
            _calculate_exponential_piecewise_<SDPA_EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_worker, scale_bf16);

        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_sum_base_idx] = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::dst_reg[prev_max_base_idx] = exp_prev;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker;
        } else {
            sfpi::vFloat curr_sum = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::vFloat recip_sum = ckernel::sfpu::sfpu_reciprocal<SDPA_EXP_APPROX_MODE>(curr_sum);
            sfpi::dst_reg[prev_max_base_idx] = exp_prev * recip_sum;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker * recip_sum;
        }
        sfpi::dst_reg += 2;
    }
}

/**
 * Wrapper for fused max-sub-exp-add SFPI kernel.
 * Invokes calculate_fused_max_sub_exp_add_tile via LLK unary SFPU parameters.
 */
template <bool SDPA_EXP_APPROX_MODE, int vector_mode = (int)VectorMode::C, bool final_norm = false>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE, final_norm>, idst, vector_mode, scale_bf16);
}
#endif

namespace ckernel {

// =============================================================================
// SDPA Tail Helpers
// =============================================================================

/**
 * Helper 1: MS Reduction Phase
 *
 * Processes MS tiles to compute P1 and P2 scaling factors, sets up SRCB for
 * subsequent L tile broadcast multiply operations.
 *
 * After this call:
 *   - SRCB contains P1 (col 0) and P2 (col 1) ready for broadcast multiply
 *   - If normalize=false: MS output is packed to cb_cur_ms, tile_regs released
 *   - If normalize=true: tile_regs still held (caller can process first L block immediately)
 *
 * @param cb_worker_ms Worker MS tile (MS1) (max in col 0, sum in col 1)
 * @param cb_prev_ms Previous MS tile (MS2) (max in col 0, sum in col 1)
 * @param cb_cur_ms Output MS tile (only used when normalize=false)
 * @param cb_l_for_init CB used for sdpa_mul_bcast_col_reuse_tiles_init
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t block_size,
    uint32_t scale_fp32,
    int vector_mode = (int)VectorMode::C,
    bool pop_ms = false>
ALWI void sdpa_tail_ms_reduce(uint32_t cb_worker_ms, uint32_t cb_prev_ms, uint32_t cb_cur_ms, uint32_t cb_l_for_init) {
    copy_tile_to_dst_init_short(cb_worker_ms);
    cb_wait_front(cb_worker_ms, 1);
    cb_wait_front(cb_prev_ms, 1);
    constexpr uint32_t dst_reg_0 = 0;  // prev_ms
    constexpr uint32_t dst_reg_1 = 1;  // worker_ms
    constexpr uint32_t dst_reg_2 = 2;  // cur_ms output

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    tile_regs_acquire();
    copy_tile(cb_prev_ms, 0, dst_reg_0);
    copy_tile(cb_worker_ms, 0, dst_reg_1);
    if constexpr (pop_ms) {
        cb_pop_front(cb_prev_ms, 1);
        cb_pop_front(cb_worker_ms, 1);
    }
    MATH((fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE, vector_mode, normalize>(0, scale_bf16)));
    // Initialize SRCB reuse for L tile broadcast multiply
    // TODO: Optimize init sequence with copy_tile
    sdpa_mul_bcast_col_reuse_tiles_init<block_size>(cb_l_for_init);
    sdpa_bcast_col_reuse_preamble<normalize>();

    // Not final reduction: pack out stats and release regs
    if constexpr (!normalize) {
        tile_regs_commit();
        cb_reserve_back(cb_cur_ms, 1);
        tile_regs_wait();
        pack_tile(dst_reg_2, cb_cur_ms);
        cb_push_back(cb_cur_ms, 1);
        tile_regs_release();
    }
}

/**
 * Helper 2: Process single L block
 *
 * Processes one block of L tiles using P1/P2 already in SRCB from sdpa_tail_ms_reduce.
 * Caller is responsible for cb_wait_front/cb_reserve_back before and cb_push_back/cb_pop_front after.
 *
 * @param cb_l1 First L input CB
 * @param cb_l2 Second L input CB
 * @param cb_l_out Output L CB
 * @param tile_index Starting tile index within the CB (for current block)
 * @param acquire_regs Whether to acquire tile_regs (false if regs already held from MS phase)
 */
template <uint32_t block_size, bool manage_cbs = false>
ALWI void sdpa_tail_l_block(uint32_t cb_l1, uint32_t cb_l2, uint32_t cb_l_out, uint32_t tile_index, bool acquire_regs) {
    if (acquire_regs) {
        tile_regs_acquire();
    }
    if constexpr (manage_cbs) {
        cb_wait_front(cb_l2, block_size);
        cb_wait_front(cb_l1, block_size);
    }
    sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, tile_index, 0);
    if constexpr (manage_cbs) {
        cb_pop_front(cb_l2, block_size);
        cb_pop_front(cb_l1, block_size);
        cb_reserve_back(cb_l_out, block_size);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, cb_l_out, block_size);
    if constexpr (manage_cbs) {
        cb_push_back(cb_l_out, block_size);
    }
    tile_regs_release();
}

/**
 * Helper 3: Finalize SDPA tail
 *
 * Cleanup: calls postamble and pops MS input tiles.
 * Call this after all L blocks have been processed.
 *
 * @param cb_worker_ms Worker MS tile CB (to pop)
 * @param cb_prev_ms Previous MS tile CB (to pop)
 */
template <bool pop_ms = true>
ALWI void sdpa_tail_finalize(uint32_t cb_worker_ms, uint32_t cb_prev_ms) {
    sdpa_bcast_col_reuse_postamble();
    if constexpr (pop_ms) {
        cb_pop_front(cb_prev_ms, 1);
        cb_pop_front(cb_worker_ms, 1);
    }
}

// =============================================================================
// SDPA Tail - Main function (uses helpers internally)
// =============================================================================

/**
 * SDPA tail reduction combining fused SFPI kernel with srcB reuse broadcast multiply.
 *
 * Implements the following reduction:
 * 1. cb_m_out = max(cb_m2, cb_m1)
 * 2. cb_exp_diff_2 = exp((cb_m1 - cb_m_out) * scale)  [P1]
 * 3. cb_s1 *= cb_exp_diff_2  (s1 * P1)
 * 4. cb_exp_diff_1 = exp((cb_m2 - cb_m_out) * scale)  [P2]
 * 5. cb_s2 *= cb_exp_diff_1  (s2 * P2)
 * 6. cb_s_out = cb_s1 + cb_s2  (s1*P1 + s2*P2)
 * 7. cb_l_out = cb_l1 * P1 + cb_l2 * P2
 *
 * @param cb_worker_max_sum Worker MS tile (MS1) (max in col 0, sum in col 1)
 * @param cb_prev_max_sum Previous MS tile (MS2) (max in col 0, sum in col 1)
 * @param cb_cur_max_sum Output MS tile (only used when normalize=false)
 * @param cb_l1 Worker L tiles
 * @param cb_l2 Previous L tiles
 * @param cb_l_out Output L tiles
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t block_size,
    uint32_t num_blocks,
    uint32_t scale_fp32,
    int vector_mode = (int)VectorMode::C>
ALWI void sdpa_tail(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out) {
    // Phase 1: MS reduction - computes P1/P2, sets up SRCB
    sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, vector_mode, true>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // Phase 2: Process all L blocks
    // When normalize=true, first block uses regs still held from MS phase
    if constexpr (normalize) {
        sdpa_tail_l_block<block_size, true>(cb_l1, cb_l2, cb_l_out, 0, false);
    }
    for (uint32_t i = (normalize ? 1 : 0); i < num_blocks; i++) {
        sdpa_tail_l_block<block_size, true>(cb_l1, cb_l2, cb_l_out, 0, true);
    }

    // Phase 3: Finalize (postamble + pop MS)
    sdpa_tail_finalize<false>(cb_worker_max_sum, cb_prev_max_sum);
}

}  // namespace ckernel
