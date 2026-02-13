// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"

#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm_reuse_dest_srcb.h"

#ifdef TRISC_PACK
#include "../../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

constexpr uint32_t SFPU_FPU = ckernel::semaphore::UNPACK_MATH_DONE;

#ifdef TRISC_PACK
// Packer:
// All 32 slots of replay buffer are used for red sum and red max
// Fast Approx Exp uses 3 constants and LoadMacro
// Non-Approx Exp uses 1 constant for recip. TODO: Look into integrating new polynomial exp in
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp

constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

template <uint32_t scale /* 1.0f in FP32 */>
inline void init_fast_approx_exp_constants() {
    constexpr float LN2_RECIP = 1.4426950408889634f;
    constexpr float A = 256.0f * LN2_RECIP;
    constexpr float B_minus_C = 32500.818359375f;
    constexpr float THRESHOLD = -88.5f;

    constexpr float scale_fp32 = __builtin_bit_cast(float, scale);

    constexpr float A_scaled = A * scale_fp32;
    constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

    TTI_SFPLOADI(0, 0xA, lo16(THRESHOLD_scaled));
    TTI_SFPLOADI(0, 0x8, hi16(THRESHOLD_scaled));
    TTI_SFPCONFIG(0, 14, 0);  // SFPCONFIG Dest 14 = LREG[14] =            -88.5               = 0xc2b10000

    TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
    TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
    TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12 = LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b

    TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
    TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
    TTI_SFPCONFIG(0, 13, 0);  // SFPCONFIG Dest 13 = LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3
}

inline void fast_approx_exp(uint32_t dst_index) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    ckernel::sfpu::calculate_exponential<true, true, DST_ACCUM_MODE, true, 4, true>();
}

// TODO: Currently hardcodes the lregs used by red max
// Could potentially also skip loading prev sum if we manage lregs properly
// TODO: Try and integrate with calculate_exponential_polynomial instead for perf
template <bool exp_approx_mode, uint16_t scale_bf16>
inline void non_approx_exp_mul_prev(uint32_t curr_sum_index, uint32_t corr_exp_index) {
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, corr_exp_index + get_dest_buffer_base());
    // Prev - Max
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, 2);  // SFPMAD_MOD1_NEGATE_VC
    TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG3, 2);  // SFPMAD_MOD1_NEGATE_VC
    sfpi::vFloat sub_top_4 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat sub_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg3];
    // Init after to avoid trampling cached registers before we use them
    // TODO: Putting the prev regs in the upper regs lets us init ahead of time
    ckernel::sfpu::_init_sfpu_reciprocal_<false>();
    sfpi::vFloat exp_top_4 = ckernel::sfpu::
        _calculate_exponential_piecewise_<exp_approx_mode, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
            sub_top_4, scale_bf16);
    sfpi::vFloat exp_bottom_4 = ckernel::sfpu::
        _calculate_exponential_piecewise_<exp_approx_mode, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
            sub_bottom_4, scale_bf16);
    // Subtract 1. This is because the bcast mul accumulates to dest
    // Without -1: bcast = prev * exp + prev
    // With -1: bcast = prev * (exp - 1) + prev = prev * exp - prev + prev = prev * exp
    sfpi::l_reg[sfpi::LRegs::LReg0] = exp_top_4;
    sfpi::l_reg[sfpi::LRegs::LReg2] = exp_bottom_4;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG1, 2);  // SFPMAD_MOD1_NEGATE_VC
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG3, 2);  // SFPMAD_MOD1_NEGATE_VC
    // Store Exp - 1 Values
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 4);
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, curr_sum_index + get_dest_buffer_base());
    // Load Curr Sum Values
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 4);
    // Mul Exp by Curr Sum
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    // Store Result
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 4);
}

// TODO: Currently hardcodes the lregs used by red max
// Could potentially also skip loading prev sum if we manage lregs properly
// TODO: Try and integrate with calculate_exponential_polynomial instead for perf
template <bool exp_approx_mode, uint16_t scale_bf16>
inline void recip_sum(uint32_t curr_sum_index, uint32_t recip_dst_index) {
    // Last op should already be sum offset
    sfpi::vFloat sum_top_4 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat sum_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg2];
    // Init after to avoid trampling cached registers before we use them
    // TODO: Putting the prev regs in the upper regs lets us init ahead of time
    ckernel::sfpu::_init_sfpu_reciprocal_<false>();
    sfpi::vFloat recip_top_4 = ckernel::sfpu::sfpu_reciprocal<exp_approx_mode>(sum_top_4);
    sfpi::vFloat recip_bottom_4 = ckernel::sfpu::sfpu_reciprocal<exp_approx_mode>(sum_bottom_4);

    // Subtract 1. This is because the bcast mul accumulates to dest
    sfpi::l_reg[sfpi::LRegs::LReg0] = recip_top_4;
    sfpi::l_reg[sfpi::LRegs::LReg2] = recip_bottom_4;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG0, 2);  // SFPMAD_MOD1_NEGATE_VC
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LCONST_1, p_sfpu::LREG2, 2);  // SFPMAD_MOD1_NEGATE_VC
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, recip_dst_index + get_dest_buffer_base());
    // Store Result
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 4);
}
#endif

// First chunk controls whether we run the correction path with prev sum, max, out
// Last chunk controls whether we signal out packer to start packing as output is produced
template <
    uint32_t chunk_size,
    uint32_t num_tiles_k,
    uint32_t num_tiles_v,
    uint32_t scale_fp32,
    uint16_t scale_bf16,
    bool transpose_k,
    bool transpose_v,
    uint32_t packed_tile_size,
    bool exp_approx_mode = false>
void compute_sdpa_chunk(
    uint32_t cb_q,
    uint32_t cb_k,
    uint32_t cb_out,
    uint32_t mm1_dst_offset,
    uint32_t mm2_dst_offset,
    uint32_t max_dst_offset,
    uint32_t sum_dst_offset,
    uint32_t corr_exp_dst_offset,
    bool first_chunk,
    bool last_chunk) {
    // TODO: This is likely needed due to a conflict with the compiler
    PACK((ckernel::sfpu::_init_sdpa_reduce_row_8x32_replay_buffers_()));
    sdpa_custom_mm_block_init_short<transpose_k>(cb_q, cb_k, cb_out, chunk_size);
    cb_wait_front(cb_k, num_tiles_k * chunk_size);
    // Q @ K (FPU)
    // Make sure SFPU of previous chunk is done (sem is zero)
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_custom_mm_block<transpose_k>(cb_q, cb_k, 0, 0, mm1_dst_offset, num_tiles_k, chunk_size);
    // Reduce Max (SFPU)
    PACK((llk_math_sfpu_sdpa_reduce_max_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size>(
        mm1_dst_offset, max_dst_offset, !first_chunk)));
    // Bcast Sub (FPU)
    // Wait for SFPU to finish (sem is 0)
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init<chunk_size>(cb_q);  // For tile shape
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(max_dst_offset);
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles<chunk_size, false>(mm1_dst_offset);
    if (!first_chunk) {
        // Exp Sub (SFPU)
        // Signal FPU that tile is ready
        // This should just init an lreg constant and is what's needed for non-approx exp
        PACK((non_approx_exp_mul_prev<exp_approx_mode, scale_bf16>(sum_dst_offset, corr_exp_dst_offset)));
        PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
        // Bcast Mul (FPU)
        // Wait for SFPU that tile is ready (sem is non-zero)
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v>(cb_q);
        MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
        sdpa_bcast_col_srca_srcb_reuse_preamble(corr_exp_dst_offset);
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, true>(mm2_dst_offset);
        // FPU has consumed the tile
        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
        // Reset to 0
        MATH((t6_semaphore_get<p_stall::MATH>(SFPU_FPU)));
    }
    // Exp Mul Scale (SFPU)
    PACK((init_fast_approx_exp_constants<scale_fp32>()));
    for (uint32_t i = 0; i < chunk_size; i++) {
        // Wait for FPU that tile is ready (sem is non-zero)
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
        PACK((fast_approx_exp(mm1_dst_offset + i * packed_tile_size)));
        PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
        PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
    }

    // MM (FPU)
    sdpa_custom_mm_reuse_dest_srcb_block_init_short(cb_q, cb_k, cb_out, transpose_v, chunk_size, num_tiles_v);
    sdpa_custom_mm_reuse_dest_srcb_block(
        cb_q,
        cb_k,
        0,
        0,
        mm1_dst_offset,
        mm2_dst_offset,
        transpose_v,
        chunk_size,
        num_tiles_v,
        num_tiles_k,
        last_chunk);

    // Reduce Sum (SFPU)
    PACK((llk_math_sfpu_sdpa_reduce_sum_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size, true>(
        mm1_dst_offset, sum_dst_offset, !first_chunk)));
    // Signal SFPU is done for the chunk
    if (!first_chunk) {
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
        PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
    }
    cb_pop_front(cb_k, num_tiles_k * chunk_size);
}

template <uint32_t num_tiles_v, bool exp_approx_mode, uint16_t scale_bf16>
void compute_sdpa_recip(uint32_t cb_q, uint32_t sum_dst_offset, uint32_t recip_dst_offset, uint32_t mm2_dst_offset) {
    PACK((recip_sum<exp_approx_mode, scale_bf16>(sum_dst_offset, recip_dst_offset)));
    PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
    sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v>(cb_q);
    MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(recip_dst_offset);
    sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, false, true>(mm2_dst_offset);
    MATH((t6_semaphore_get<p_stall::MATH>(SFPU_FPU)));
}
