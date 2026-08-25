// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/sdpa_custom_mm.h"
#include "api/compute/experimental/sdpa_custom_mm_reuse_dest_srcb.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_sdpa_bcast_col_srcb_reuse_api.h"
#include "experimental/llk_math_sdpa_bcast_col_srca_srcb_reuse_api.h"
#include "experimental/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "experimental/llk_sfpu/ckernel_sfpu_deepseek_sdpa.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_A_sdpa_api.h"
#endif
#ifdef TRISC_PACK
#include "experimental/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    std::uint32_t num_tiles,
    bool dense = false>
ALWI void sdpa_bcast_col_reuse_tiles_init(std::uint32_t icb0) {
    UNPACK((llk_unpack_A_sdpa_init<num_tiles, BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(
        false, false, icb0)));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse_init_with_operands<eltwise_binary_type, num_tiles, MATH_FIDELITY, dense>(
        icb0, icb0, false)));
}

template <bool clear_dest = false>
ALWI void sdpa_bcast_col_reuse_preamble() {
    UNPACK((llk_unpack_A_sdpa_set_srcb_dummy_valid()));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse_preamble<DST_SYNC_MODE, DST_ACCUM_MODE, clear_dest>()));
}

ALWI void sdpa_bcast_col_reuse_postamble() { MATH((llk_math_sdpa_bcast_col_srcb_reuse_postamble())); }

template <EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD, std::uint32_t num_tiles>
ALWI void sdpa_bcast_col_reuse_tiles(
    std::uint32_t in0_cb_id, std::uint32_t in1_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(in0_cb_id, in_tile_index)));
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(in1_cb_id, in_tile_index)));
    MATH((llk_math_sdpa_bcast_col_srcb_reuse<eltwise_binary_type, num_tiles, DST_ACCUM_MODE, MATH_FIDELITY>(
        dst_tile_index)));
}

template <std::uint32_t num_tiles, bool dense = false>
ALWI void sdpa_mul_bcast_col_reuse_tiles_init(std::uint32_t icb0) {
    sdpa_bcast_col_reuse_tiles_init<EltwiseBinaryType::ELWMUL, num_tiles, dense>(icb0);
}

template <std::uint32_t num_tiles>
ALWI void sdpa_mul_bcast_col_reuse_tiles(
    std::uint32_t in0_cb_id, std::uint32_t in1_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    sdpa_bcast_col_reuse_tiles<EltwiseBinaryType::ELWMUL, num_tiles>(
        in0_cb_id, in1_cb_id, in_tile_index, dst_tile_index);
}

template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    std::uint32_t num_tiles,
    bool skip_addrmod = false>
ALWI void sdpa_bcast_col_srca_srcb_reuse_tiles_init(std::uint32_t icb0) {
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse_init_with_operands<
          eltwise_binary_type,
          num_tiles,
          MATH_FIDELITY,
          skip_addrmod>(icb0, icb0, false)));
}

template <bool clear_dest = false>
ALWI void sdpa_bcast_col_srca_srcb_reuse_preamble(std::uint32_t isrc) {
    UNPACK((llk_unpack_A_sdpa_set_srca_srcb_dummy_valid()));
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble<DST_SYNC_MODE, DST_ACCUM_MODE, clear_dest>(isrc)));
}

template <
    EltwiseBinaryType eltwise_binary_type = EltwiseBinaryType::ELWADD,
    std::uint32_t num_tiles,
    bool skip_signalling = false,
    std::uint32_t output_granularity>
ALWI void sdpa_bcast_col_srca_srcb_reuse_tiles(std::uint32_t dst_tile_index) {
    MATH((llk_math_sdpa_bcast_col_srca_srcb_reuse<
          eltwise_binary_type,
          num_tiles,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          skip_signalling,
          output_granularity>(dst_tile_index)));
}

template <std::uint32_t num_tiles>
ALWI void sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init(std::uint32_t icb0) {
    sdpa_bcast_col_srca_srcb_reuse_tiles_init<EltwiseBinaryType::ELWSUB, num_tiles>(icb0);
}

template <std::uint32_t num_tiles, bool skip_signalling = false, std::uint32_t output_granularity>
ALWI void sdpa_sub_bcast_col_srca_srcb_reuse_tiles(std::uint32_t dst_tile_index) {
    sdpa_bcast_col_srca_srcb_reuse_tiles<EltwiseBinaryType::ELWSUB, num_tiles, skip_signalling, output_granularity>(
        dst_tile_index);
}

template <std::uint32_t num_tiles, bool skip_addrmod = false>
ALWI void sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init(std::uint32_t icb0) {
    sdpa_bcast_col_srca_srcb_reuse_tiles_init<EltwiseBinaryType::ELWMUL, num_tiles, skip_addrmod>(icb0);
}

template <std::uint32_t num_tiles, bool skip_signalling = false, std::uint32_t output_granularity>
ALWI void sdpa_mul_bcast_col_srca_srcb_reuse_tiles(std::uint32_t dst_tile_index) {
    sdpa_bcast_col_srca_srcb_reuse_tiles<EltwiseBinaryType::ELWMUL, num_tiles, skip_signalling, output_granularity>(
        dst_tile_index);
}

template <DataFormat format>
ALWI void sdpa_reduce_row_init() {
    MATH((llk_math_sfpu_sdpa_reduce_row_init<APPROX, DST_ACCUM_MODE, format>()));
}

template <DataFormat format, std::uint32_t block_width>
ALWI void sdpa_reduce_max_row(std::uint32_t src_index, std::uint32_t dst_index, bool prev_max = false) {
    MATH((llk_math_sfpu_sdpa_reduce_max_row<APPROX, DST_ACCUM_MODE, format, block_width>(
        src_index, dst_index, prev_max)));
}

template <DataFormat format, std::uint32_t block_width>
ALWI void sdpa_reduce_sum_row(std::uint32_t src_index, std::uint32_t dst_index, bool prev_sum = false) {
    MATH((llk_math_sfpu_sdpa_reduce_sum_row<APPROX, DST_ACCUM_MODE, format, block_width>(
        src_index, dst_index, prev_sum)));
}

#ifdef TRISC_PACK
// Packer:
// Fast Approx Exp uses 3 constants and LoadMacro
// Non-Approx Exp uses 1 constant for recip. TODO: Look into integrating new polynomial exp in
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp

// TODO: Factor this out into a reusable fn in LLK
template <std::uint32_t scale /* 1.0f in FP32 */>
inline void init_fast_approx_exp_constants() {
    constexpr float LN2_RECIP = 1.4426950408889634f;
    constexpr float A = 256.0f * LN2_RECIP;
    constexpr float B_minus_C = 32500.818359375f;
    constexpr float THRESHOLD = -88.5f;

    constexpr float scale_fp32 = __builtin_bit_cast(float, scale);

    constexpr float A_scaled = A * scale_fp32;
    constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

    TTI_SFPLOADI(0, 0xA, sfpu::lo16(THRESHOLD_scaled));
    TTI_SFPLOADI(0, 0x8, sfpu::hi16(THRESHOLD_scaled));
    TTI_SFPCONFIG(0, 14, 0);  // SFPCONFIG Dest 14 = LREG[14] =            -88.5               = 0xc2b10000

    TTI_SFPLOADI(0, 0xA, sfpu::lo16(A_scaled));
    TTI_SFPLOADI(0, 0x8, sfpu::hi16(A_scaled));
    TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12 = LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b

    TTI_SFPLOADI(0, 0xA, sfpu::lo16(B_minus_C));
    TTI_SFPLOADI(0, 0x8, sfpu::hi16(B_minus_C));
    TTI_SFPCONFIG(0, 13, 0);  // SFPCONFIG Dest 13 = LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3
}

inline void fast_approx_exp(std::uint32_t dst_index) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    ckernel::sfpu::calculate_exponential<true, DST_ACCUM_MODE, true, 8, true>();
}

// TODO: Currently hardcodes the lregs used by red max
// Could potentially also skip loading prev sum if we manage lregs properly
// TODO: Try and integrate with calculate_exponential_polynomial instead for perf
template <bool exp_approx_mode, std::uint16_t scale_bf16>
inline void non_approx_exp_mul_prev(std::uint32_t curr_sum_index, std::uint32_t corr_exp_index) {
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, corr_exp_index + get_dest_buffer_base());
    sfpi::vFloat prev_max_top_4 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat prev_max_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg3];
    sfpi::vFloat curr_max_top_4 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat curr_max_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat sub_top_4 = prev_max_top_4 - curr_max_top_4;
    sfpi::vFloat sub_bottom_4 = prev_max_bottom_4 - curr_max_bottom_4;
    sfpi::vFloat exp_top_4 =
        sfpu::_ckernel_sfpu_exp_accurate_upper_unclamped_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
            sub_top_4, scale_bf16);
    sfpi::vFloat exp_bottom_4 =
        sfpu::_ckernel_sfpu_exp_accurate_upper_unclamped_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
            sub_bottom_4, scale_bf16);
    // Subtract 1. This is because the bcast mul accumulates to dest
    // Without -1: bcast = prev * exp + prev
    // With -1: bcast = prev * (exp - 1) + prev = prev * exp
    dst_reg[0] = exp_top_4 - 1.0f;
    dst_reg[2] = exp_bottom_4 - 1.0f;
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, curr_sum_index + get_dest_buffer_base());
    // Load Curr Sum Values
    sfpi::vFloat curr_sum_top_4 = dst_reg[0];
    sfpi::vFloat curr_sum_bottom_4 = dst_reg[2];
    sfpi::vFloat mul_top_4 = curr_sum_top_4 * exp_top_4;
    sfpi::vFloat mul_bottom_4 = curr_sum_bottom_4 * exp_bottom_4;
    dst_reg[0] = mul_top_4;
    dst_reg[2] = mul_bottom_4;
}

// TODO: Currently hardcodes the lregs used by red max
// Could potentially also skip loading prev sum if we manage lregs properly
// TODO: Try and integrate with calculate_exponential_polynomial instead for perf
template <bool exp_approx_mode, std::uint16_t scale_bf16>
inline void recip_sum(std::uint32_t curr_sum_index, std::uint32_t recip_dst_index) {
    // Last op should already be sum offset
    sfpi::vFloat sum_top_4 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat sum_bottom_4 = sfpi::l_reg[sfpi::LRegs::LReg2];
    // Init after to avoid trampling cached registers before we use them
    // TODO: Putting the prev regs in the upper regs lets us init ahead of time
    ckernel::sfpu::sfpu_reciprocal_init<false>();
    sfpi::vFloat recip_top_4 = ckernel::sfpu::sfpu_reciprocal<exp_approx_mode>(sum_top_4);
    sfpi::vFloat recip_bottom_4 = ckernel::sfpu::sfpu_reciprocal<exp_approx_mode>(sum_bottom_4);

    // Subtract 1. This is because the bcast mul accumulates to dest
    // TODO: Can get rid of this
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, recip_dst_index + get_dest_buffer_base());
    dst_reg[0] = recip_top_4 - 1.0f;
    dst_reg[2] = recip_bottom_4 - 1.0f;
}
#endif

// First chunk controls whether we run the correction path with prev sum, max, out
// Last chunk controls whether we signal out packer to start packing as output is produced
//
// output_granularity controls how often the QK^T*V matmul (sdpa_custom_mm_reuse_dest_srcb_block)
// signals the packer via the FPU->SFPU semaphore. The packer must consume tiles in matching
// groups of output_granularity. num_tiles_v must be divisible by output_granularity.
template <
    std::uint32_t chunk_size,
    std::uint32_t num_tiles_k,
    std::uint32_t num_tiles_v,
    std::uint32_t scale_fp32,
    bool transpose_k,
    bool transpose_v,
    std::uint32_t packed_tile_size,
    bool exp_approx_mode = false,
    // qk_signal_granularity: number of QK-matmul tiles produced per FPU->SFPU (QK->reduce-max) signal.
    std::uint32_t qk_signal_granularity = 1,
    // exp_signal_granularity: number of exp'd score tiles produced per SFPU->FPU (exp->OV) signal.
    std::uint32_t exp_signal_granularity = 1,
    // output_granularity = number of O·V output tiles produced per FPU->SFPU signal.
    std::uint32_t output_granularity = 1,
    bool mm_pack_init = true,
    // separate_v=false: single-CB (MLA) layout — V is read from cb_k at stride num_tiles_k.
    // separate_v=true:  two-CB (GQA) layout — V is read from its own cb_v at stride num_tiles_v
    //                   (V tiles contiguous in cb_v), and cb_v is waited on / popped separately.
    bool separate_v = false>
void compute_sdpa_chunk(
    std::uint32_t cb_q,
    std::uint32_t cb_k,
    std::uint32_t cb_v,
    std::uint32_t cb_mask,
    std::uint32_t cb_out,
    std::uint32_t mm1_dst_offset,
    std::uint32_t mm2_dst_offset,
    std::uint32_t max_dst_offset,
    std::uint32_t sum_dst_offset,
    std::uint32_t corr_exp_dst_offset,
    bool first_chunk,
    bool last_chunk,
    bool mask_chunk,
    std::uint32_t ov_kt_dim = chunk_size) {
    constexpr std::uint16_t scale_bf16 = scale_fp32 >> 16;
    static_assert(DST_ACCUM_MODE == false, "compute_sdpa_chunk: FP32 destination accumulation mode is not supported");
    static_assert(
        num_tiles_v % output_granularity == 0,
        "compute_sdpa_chunk: num_tiles_v must be divisible by output_granularity");
    static_assert(
        chunk_size % qk_signal_granularity == 0,
        "compute_sdpa_chunk: chunk_size must be divisible by qk_signal_granularity");
    // The custom-MM unpacks issue a single MOP of (kt_dim/2)-1 iterations and only support an even
    // kt_dim >= 2 (see the "kt_dim: even number from 2 to 256" note in the unpack LLK headers). With
    // kt_dim < 2 the count underflows to ~4 billion and the unpacker hangs silently. Fail at compile
    // time instead. OV matmul: kt_dim = chunk_size (Sk_chunk_t); QK matmul: kt_dim = num_tiles_k (DHt).
    static_assert(
        chunk_size >= 2 && chunk_size % 2 == 0,
        "compute_sdpa_chunk: chunk_size (Sk_chunk_t = OV-matmul kt_dim) must be even and >= 2; "
        "k_chunk_size of 32 gives Sk_chunk_t=1, which underflows the OV custom-MM unpack MOP and hangs");
    static_assert(
        num_tiles_k >= 2 && num_tiles_k % 2 == 0,
        "compute_sdpa_chunk: num_tiles_k (QK-matmul kt_dim) must be even and >= 2 for the custom-MM unpack");
    static_assert(chunk_size + 1 <= 15, "compute_sdpa_chunk: chunk_size + 1 must be with tensix semaphore range");
    static_assert(
        num_tiles_v / output_granularity <= 15,
        "compute_sdpa_chunk: num_tiles_v / output_granularity must be with tensix semaphore range");
    PACK((ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_()));
    sdpa_custom_mm_block_init_short<transpose_k, mm_pack_init>(cb_q, cb_k, cb_out, chunk_size);
    cb_wait_front(cb_k, num_tiles_k * chunk_size);
    // Q @ K (FPU)
    // Make sure SFPU of previous chunk is done (sem is zero)
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_custom_mm_block<transpose_k, qk_signal_granularity>(
        cb_q, cb_k, cb_mask, 0, 0, mm1_dst_offset, num_tiles_k, chunk_size, mask_chunk);

    // Reduce Max (SFPU)
    PACK((llk_math_sfpu_sdpa_reduce_max_row<
          false,
          DST_ACCUM_MODE,
          DataFormat::Float16_b,
          chunk_size,
          false,
          qk_signal_granularity>(mm1_dst_offset, max_dst_offset, !first_chunk)));
    // Bcast Sub (FPU)
    // Wait for SFPU to finish (sem is 0)
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init<chunk_size>(cb_q);  // For tile shape
    MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(max_dst_offset);
    sdpa_sub_bcast_col_srca_srcb_reuse_tiles<chunk_size, false, exp_signal_granularity>(mm1_dst_offset);
    if (!first_chunk) {
        // Exp Sub (SFPU)
        // Signal FPU that tile is ready
        // This should just init an lreg constant and is what's needed for non-approx exp
        PACK((non_approx_exp_mul_prev<exp_approx_mode, scale_bf16>(sum_dst_offset, corr_exp_dst_offset)));
        PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
        // Bcast Mul (FPU)
        // Wait for SFPU that tile is ready (sem is non-zero)
#ifdef TRISC_MATH
        constexpr bool skip_addrmod = !is_high_fidelity(MATH_FIDELITY);
#else
        constexpr bool skip_addrmod = false;
#endif
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v, skip_addrmod>(cb_q);
        MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
        sdpa_bcast_col_srca_srcb_reuse_preamble(corr_exp_dst_offset);
        sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, true, 1>(mm2_dst_offset);
        // FPU has consumed the tile
        MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
        // Reset to 0
        // No stall since we stalled math already
        MATH((t6_semaphore_get<p_stall::NONE>(SFPU_FPU)));
    } else {
        // Dummy inc used so we can decrement to signal SFPU is done the current chunk
        MATH((t6_semaphore_post<p_stall::NONE>(semaphore::FPU_SFPU)));
    }
    // Exp Mul Scale (SFPU)
    static_assert(
        chunk_size % exp_signal_granularity == 0,
        "compute_sdpa_chunk: chunk_size must be divisible by exp_signal_granularity");
    for (std::uint32_t i = 0; i < chunk_size; i++) {
        if (i % exp_signal_granularity == 0) {
            PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
        }
        // Each tile is 8x32, which is the same as a full 16x16 face
        PACK((fast_approx_exp(mm1_dst_offset + i * packed_tile_size)));
        if (i % exp_signal_granularity == exp_signal_granularity - 1) {
            PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
            PACK((t6_semaphore_post<p_stall::NONE>(SFPU_FPU)));
        }
    }

    // MM (FPU). Single-CB (MLA): V is read from cb_k at stride num_tiles_k. Two-CB (GQA):
    // V lives in its own cb_v (contiguous, stride num_tiles_v) and must be waited on
    // separately. separate_v is compile-time, so ov_cb / in1_k_stride fold to constants.
    if constexpr (separate_v) {
        cb_wait_front(cb_v, num_tiles_v * chunk_size);
    }
    const std::uint32_t ov_cb = separate_v ? cb_v : cb_k;
    constexpr std::uint32_t in1_k_stride = separate_v ? num_tiles_v : num_tiles_k;
    // OV-trim (ov_kt_dim < chunk_size on the masked last chunk) keeps the exp->OV SFPU_FPU handshake
    // balanced ONLY when the whole chunk is a single signal group
    // TODO: support finer granularity via an aggregate end-of-remainder drain if needed.
    MATH((LLK_ASSERT(
        ov_kt_dim == chunk_size || exp_signal_granularity == chunk_size,
        "OV-trim (ov_kt_dim != chunk_size) requires exp_signal_granularity == chunk_size")));
    sdpa_custom_mm_reuse_dest_srcb_block_init_short(cb_q, ov_cb, cb_out, transpose_v, chunk_size, num_tiles_v);
    sdpa_custom_mm_reuse_dest_srcb_block<output_granularity, exp_signal_granularity>(
        cb_q,
        ov_cb,
        0,
        0,
        mm1_dst_offset,
        mm2_dst_offset,
        transpose_v,
        ov_kt_dim,
        num_tiles_v,
        in1_k_stride,
        last_chunk);

    // Reduce Sum (SFPU)
    PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
    PACK((llk_math_sfpu_sdpa_reduce_sum_row<false, DST_ACCUM_MODE, DataFormat::Float16_b, chunk_size, true>(
        mm1_dst_offset, sum_dst_offset, !first_chunk)));
    // Signal SFPU is done for the chunk (so QK MM can reuse the space next iteration)
    PACK((llk_math_sdpa_sfpu_signal_chunk_done()));
    cb_pop_front(cb_k, num_tiles_k * chunk_size);
    if constexpr (separate_v) {
        cb_pop_front(cb_v, num_tiles_v * chunk_size);
    }
}

template <std::uint32_t num_tiles_v, bool exp_approx_mode, std::uint32_t scale_fp32, std::uint32_t output_granularity>
void compute_sdpa_recip(
    std::uint32_t cb_q, std::uint32_t sum_dst_offset, std::uint32_t recip_dst_offset, std::uint32_t mm2_dst_offset) {
    constexpr std::uint16_t scale_bf16 = scale_fp32 >> 16;
    PACK((recip_sum<exp_approx_mode, scale_bf16>(sum_dst_offset, recip_dst_offset)));
    PACK((t6_semaphore_post<p_stall::WAIT_SFPU>(SFPU_FPU)));
    sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v>(cb_q);
    MATH((t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)));
    sdpa_bcast_col_srca_srcb_reuse_preamble(recip_dst_offset);
    sdpa_mul_bcast_col_srca_srcb_reuse_tiles<num_tiles_v, false, output_granularity>(mm2_dst_offset);
    MATH((t6_semaphore_get<p_stall::MATH>(SFPU_FPU)));
}

// =============================================================================
// SDPA Tail Reduction - Fused SFPI Kernel and Helper
// =============================================================================

#ifdef TRISC_MATH

/**
 * Wrapper for fused max-sub-exp-add SFPI kernel.
 * Invokes calculate_fused_max_sub_exp_add_tile via the SFPU macro wrapper.
 */
template <bool SDPA_EXP_APPROX_MODE, VectorMode vector_mode = VectorMode::C, bool final_norm = false>
void fused_max_sub_exp_add_tile(std::uint32_t idst, int scale_bf16) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_fused_max_sub_exp_add_tile,
        (SDPA_EXP_APPROX_MODE, final_norm),
        idst,
        vector_mode,
        scale_bf16);
}
#endif

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
    std::uint32_t block_size,
    std::uint32_t scale_fp32,
    VectorMode vector_mode = VectorMode::C,
    bool pop_ms = false,
    bool dense = false>
ALWI void sdpa_tail_ms_reduce(
    std::uint32_t cb_worker_ms, std::uint32_t cb_prev_ms, std::uint32_t cb_cur_ms, std::uint32_t cb_l_for_init) {
    copy_init(cb_worker_ms);
    cb_wait_front(cb_worker_ms, 1);
    cb_wait_front(cb_prev_ms, 1);
    constexpr std::uint32_t dst_reg_0 = 0;  // prev_ms
    constexpr std::uint32_t dst_reg_1 = 1;  // worker_ms
    constexpr std::uint32_t dst_reg_2 = 2;  // cur_ms output

    constexpr std::uint16_t scale_bf16 = scale_fp32 >> 16;

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
    sdpa_mul_bcast_col_reuse_tiles_init<block_size, dense>(cb_l_for_init);
    sdpa_bcast_col_reuse_preamble<normalize>();

    // Not final reduction: pack out stats and release regs
    if constexpr (!normalize) {
        PACK((llk_pack_init<
              ckernel::PackMode::Default /* untilize=false, tilize=false */,
              false /* zero_output */,
              true /* skip_addrmod_config */>(cb_cur_ms)));
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
template <
    std::uint32_t block_size,
    std::uint32_t num_blocks,
    bool untilize = false,
    bool dense = false,
    bool manage_cbs = false>
ALWI void sdpa_tail_l_block(
    std::uint32_t cb_l1,
    std::uint32_t cb_l2,
    std::uint32_t cb_l_out,
    std::uint32_t tile_index,
    std::uint32_t block_index,
    bool acquire_regs) {
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
        if constexpr (!untilize) {
            cb_reserve_back(cb_l_out, block_size);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    if constexpr (untilize) {
        pack_untilize_dest<block_size, block_size * num_blocks, false, false, TILE_C_DIM, 0, dense>(
            cb_l_out, 1, block_index);
    } else {
        pack_block_contiguous(0, cb_l_out, block_size);
    }
    if constexpr (manage_cbs) {
        if constexpr (!untilize) {
            cb_push_back(cb_l_out, block_size);
        }
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
ALWI void sdpa_tail_finalize(std::uint32_t cb_worker_ms, std::uint32_t cb_prev_ms) {
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
    std::uint32_t block_size,
    std::uint32_t num_blocks,
    std::uint32_t scale_fp32,
    VectorMode vector_mode = VectorMode::C,
    bool dense = false,
    bool untilize = false>
ALWI void sdpa_tail(
    std::uint32_t cb_worker_max_sum,
    std::uint32_t cb_prev_max_sum,
    std::uint32_t cb_cur_max_sum,
    std::uint32_t cb_l1,
    std::uint32_t cb_l2,
    std::uint32_t cb_l_out) {
    // Phase 1: MS reduction - computes P1/P2, sets up SRCB
    sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, vector_mode, true, dense>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // TODO: Update the tile locs in ms_reduce to enable dense packing during entire reduction
    if constexpr (dense && !untilize) {
        // Reduce packing stride from tile to tile to 32 rows instead of 64
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(
            (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2)));
    }
    if constexpr (!untilize) {
        pack_block_contiguous_init(cb_l_out);
    }

    // Phase 2: Process all L blocks
    // Untilize requires operating on all blocks at once
    if constexpr (untilize) {
        // Canonical pack_untilize with configure_remap=false: skips the MATH remap reconfig, which
        // clobbers state when this fused block runs after another compute op on the same core. The
        // face geometry (8-row faces, 2/4 faces) is derived from cb_l_out's CB configuration.
        pack_untilize_dest_init<
            block_size,
            num_blocks * block_size,
            false /*narrow_row*/,
            TILE_C_DIM,
            dense,
            false /*configure_remap*/>(cb_l_out);
        cb_reserve_back(cb_l_out, block_size * num_blocks);
    }
    // When normalize=true, first block uses regs still held from MS phase
    if constexpr (normalize) {
        sdpa_tail_l_block<block_size, num_blocks, untilize, dense, true>(cb_l1, cb_l2, cb_l_out, 0, 0, false);
    }
    for (std::uint32_t i = (normalize ? 1 : 0); i < num_blocks; i++) {
        sdpa_tail_l_block<block_size, num_blocks, untilize, dense, true>(cb_l1, cb_l2, cb_l_out, 0, i, true);
    }
    if constexpr (untilize) {
        cb_push_back(cb_l_out, block_size * num_blocks);
        pack_untilize_uninit(cb_l_out);
    }

    if constexpr (dense && !untilize) {
        // Restore packing stride from tile to tile to 64 rows
        PACK((cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2)));
    }

    // Phase 3: Finalize (postamble + pop MS)
    sdpa_tail_finalize<false>(cb_worker_max_sum, cb_prev_max_sum);
}

}  // namespace ckernel
