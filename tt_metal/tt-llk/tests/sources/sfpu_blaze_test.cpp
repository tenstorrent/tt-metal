// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lane FD blaze-registration vehicle: drives the vendored tt-blaze SFPU
// kernels (helpers/include/blaze_vendored/, see VENDORED.md there) on the
// standard corpus harness so the blaze bodies get real 2x2 numbers on our
// sweep.  The vendored originals are byte-exact copies of tt-blaze
// nkapre/sfpi @ 69b8782e2; the lifts are lanes EW/EX's typed semantic
// bodies from the same commit.  All wiring is test-side (LLK-pristine R7).
//
// BLAZE_OP selects the kernel (values mirror test_sfpu_blaze.py::BlazeOp):
//   1 clampedsilu_gate   blaze sfpu/clamped_silu_sfpu.hpp  (typed; sem == orig)
//   2 clampedsilu_up     "
//   3 clampedsilu_clamped "
//   4 situ_gate          "
//   5 scaledtanh         "
//   6 logitsoftcap       blaze semantic/logit_softcap.hpp MATH-gate twin
//                        (the original is #ifdef TRISC_PACK-only; the twin is
//                        its byte-equivalent body under a MATH||PACK gate)
//   7 siluscaled         blaze sfpu/silu_scaled.hpp
//   8 sparsekfilter      blaze sfpu/sparse_k_filter_sfpu.hpp (Int32)
//   9 zeropad            blaze sfpu/zero_pad_sfpu.hpp
//  10 addrsqrt           blaze kernel_includes .../experimental add_rsqrt
//  11 sdpaexp            blaze kernel_includes .../experimental sdpa_exp_unclamped
//  12 rope               blaze RAW-TTI rope        vs lane-EW lift (BLAZE_IMPL)
//  13 sdpareducerow      blaze RAW-TTI reduce_row  vs lane-EW lift (BLAZE_IMPL,
//                        BLAZE_SUBOP: 0 MAX / 1 SUM)
// BLAZE_IMPL: 0 = byte-exact vendored blaze original (hand arm),
//             2 = vendored typed semantic lift        (sem arm).
// Ops 1-11 are typed already-semantic sources (sem == orig): BLAZE_IMPL is 0.
// BLAZE_PARAM0/BLAZE_PARAM1: per-op scalar bits (see BLAZE_PARAMS in
// helpers/test_variant_parameters.py).
//
// Lane FE multi-tile extension: the vehicle drives params.TILE_CNT tiles per
// run (the softmax_k-vehicle runtime TILE_COUNT mechanism).  Per-op inits are
// hoisted BEFORE the tile loop (run once per kernel — the amortization under
// measurement); the loop body is wait-dest / datacopy / BLAZE_BODY zone /
// section-done per tile.  The BLAZE_BODY zone stays per-tile INSIDE the loop,
// so its profiled figure is the mean per-tile body time at every TILE_CNT
// (the report's _stats_timings takes the mean across zone instances) and
// cells stay directly comparable across tile counts.  rope re-unpacks the
// single cos/sin tile (buffer_B[0]) every iteration.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "counters.h"
#include "llk_defs.h"
// params.h FIRST (build.h carries the BLAZE_* template defines; it must
// precede the #ifndef defaults below — the coverage-vehicle include order).
#include "params.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifndef BLAZE_OP
#define BLAZE_OP 1
#endif
#ifndef BLAZE_SUBOP
#define BLAZE_SUBOP 0
#endif
#ifndef BLAZE_IMPL
#define BLAZE_IMPL 0
#endif

// rope stages cos into Dst tile 1's faces 0-1 and sin into its faces 2-3.
static constexpr bool BLAZE_TWO_TILE = (BLAZE_OP == 12);

// Fixed sparse-k-filter field geometry (the laneEU coverage-vehicle values),
// mirrored by the python golden.
static constexpr std::uint32_t BSKF_BANK_MASK         = 0x3F;
static constexpr std::uint32_t BSKF_MY_BANK           = 5;
static constexpr std::uint32_t BSKF_GLOBAL_BANK_SHIFT = 10;
static constexpr std::uint32_t BSKF_WITHIN_BANK_MASK  = 0x3FF;
static constexpr std::uint32_t BSKF_OUT_SHIFT         = 0;

// Fixed zero-pad row split, mirrored by the python golden.
static constexpr int BZP_VALID_ROWS = 24;
static constexpr int BZP_TOTAL_ROWS = 32;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t UNPACK_FMT = UNPACK_A_IN;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), UNPACK_FMT, UNPACK_FMT);
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[tile]), UNPACK_FMT, UNPACK_FMT);
        if constexpr (BLAZE_TWO_TILE)
        {
            // The single cos/sin tile is shared by every x tile.
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_B[0]), UNPACK_FMT, UNPACK_FMT);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

// The vendored blaze bodies gate on the tt-metal JIT thread define
// (TRISC_MATH), not the tt-llk harness's LLK_TRISC_MATH.  This TU is the
// math thread, so the tt-metal spelling is true here (test-side shim; the
// vendored files stay byte-exact).
#ifndef TRISC_MATH
#define TRISC_MATH 1
#endif

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

// SFPU call/init macro layer FIRST (forward-declares production inits;
// -Wredundant-decls requires declarations to precede definitions).
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
// Converter + helpers the vendored kernels expect in scope.  The blaze JIT
// include order resolves bare "ckernel_sfpu_tanh.h"/"ckernel_sfpu_exp.h" to
// the metal llk_sfpu tree; here the tt-llk copy wins the bare spelling, so
// pull the metal helpers explicitly first (harness-side, R7).
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_converter.h"
// Byte-exact vendored blaze kernels (originals + lifts); their in-repo
// "blaze/kernels/..." spellings resolve via -Ihelpers/include/blaze_vendored.
#include "blaze/kernels/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_rope.h"
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h"
#include "blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_reduce_row.h"
#include "blaze/kernels/sfpu/clamped_silu_sfpu.hpp"
#include "blaze/kernels/sfpu/semantic/logit_softcap.hpp"
#include "blaze/kernels/sfpu/semantic/rope.hpp"
#include "blaze/kernels/sfpu/semantic/sdpa_reduce_row.hpp"
#include "blaze/kernels/sfpu/semantic/sdpa_reduce_row_crosslane.hpp"
#include "blaze/kernels/sfpu/semantic/sdpa_reduce_row_walk.hpp"
#include "blaze/kernels/sfpu/silu_scaled.hpp"
#include "blaze/kernels/sfpu/sparse_k_filter_sfpu.hpp"
#include "blaze/kernels/sfpu/zero_pad_sfpu.hpp"
#include "blaze_twins/sdpa_reduce_row_uniform.hpp" // lane-IE uniform twin (test-side, outside the vendored root — R7)

// Per-op init frame (shared by both arms; protocol, not the raced math).
static inline void blaze_op_init()
{
#if BLAZE_OP == 1 || BLAZE_OP == 4 || BLAZE_OP == 5 || BLAZE_OP == 7
    // These bodies evaluate _sfpu_sigmoid_, whose internal reciprocal reads
    // the recip programmable constants — blaze's callers run silu/sigmoid
    // init first (the silu_scaled header says so explicitly), so the
    // production init is protocol here.  Without it the kernel silently
    // depends on ambient PRGM state (caught by cross-node sim-state
    // contamination at wiring).
    sfpu::sigmoid_init<false /* APPROXIMATION_MODE */>();
#elif BLAZE_OP == 6
    // _sfpu_tanh_fp32_accurate_ (the twin's helper) reads the accurate-tanh
    // programmable constants; program them exactly as the production tanh
    // init's accurate arm does (blaze's PACK-thread caller owns this setup
    // in-repo, so it is protocol here, not raced math).
    sfpi::vConstFloatPrgm0 = 2.0f * 1.442695f;     // 2 * log2(e)
    sfpi::vConstFloatPrgm1 = -0.6931471805599453f; // -ln(2)
    sfpi::vConstFloatPrgm2 = 1.666667163e-1f;      // expm1 c1
#elif BLAZE_OP == 10
    sfpu::init_add_rsqrt<APPROX_MODE>();
#elif BLAZE_OP == 11
    // The sdpa exp helper shares production exp's programmable-constant setup.
    sfpu::exp_init<false /* APPROX */, 0x3F800000, true /* CLAMP_NEGATIVE */, is_fp32_dest_acc_en>();
#elif BLAZE_OP == 12
    sfpu::sfpu_rope_configure_addrmod();
    sfpu::sfpu_rope_dest_setup();
#elif BLAZE_OP == 13
    sfpu::_init_sdpa_reduce_row_8x32_<DataFormat::Float16_b>();
#if BLAZE_SUBOP == 0
    sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_();
#else
    sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_();
#endif
#endif
}

// Wrapper loop for the value-function kernel (op 11): plain typed drive of
// the vendored value function over the full tile, one row per iteration.
// Guarded by op: the vendored helper static_asserts a bf16 dest, so it must
// not be instantiated in the Int32/dest-acc TUs of the other ops.
#if BLAZE_OP == 11
namespace ckernel
{
namespace sfpu
{
inline void blaze_sdpa_exp_tile()
{
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++)
    {
        sfpi::vFloat x   = sfpi::dst_reg[0];
        sfpi::vFloat y   = _ckernel_sfpu_exp_accurate_upper_unclamped_<false /* SCALE_EN */, is_fp32_dest_acc_en>(x, 0);
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
} // namespace sfpu
} // namespace ckernel
#endif // BLAZE_OP == 11

void run_kernel(RUNTIME_PARAMETERS params)
{
    const bool is_int_fpu_en    = false;
    const std::uint8_t MATH_FMT = UNPACK_A_IN;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        4 /* num_faces */, MATH_FMT);

    // Per-op inits hoisted before the tile loop: they program SFPU config /
    // PRGM constants / replay buffers, never Dst data, so they are legal
    // before the first wait-for-dest — and running them ONCE per kernel is
    // the fixed cost whose amortization the multi-tile rows measure.
    SFPU_UNARY_INIT(unused);
    blaze_op_init();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, MATH_FMT, MATH_FMT);
        if constexpr (BLAZE_TWO_TILE)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                1 /* dst_index */, MATH_FMT, MATH_FMT);
        }
        // Reset dest addressing before the SFPU op (coverage-vehicle convention).
        _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

        {
            // Named device-profile zone (test_sfpu_blaze.py::*_device_profile).
            START_PERF_MEASURE("BLAZE_BODY")

#if BLAZE_OP == 1
        SFPU_UNARY_CALL(
            DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_clamped_silu_gate, (is_fp32_dest_acc_en, 32), 0, VectorMode::None, BLAZE_PARAM0, BLAZE_PARAM1);
#elif BLAZE_OP == 2
        SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_clamped_up, (is_fp32_dest_acc_en, 32), 0, VectorMode::None, BLAZE_PARAM0);
#elif BLAZE_OP == 3
        SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_clamped, (is_fp32_dest_acc_en, 32), 0, VectorMode::None, BLAZE_PARAM0);
#elif BLAZE_OP == 4
        SFPU_UNARY_CALL(
            DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_situ_gate, (is_fp32_dest_acc_en, 32), 0, VectorMode::None, BLAZE_PARAM0, BLAZE_PARAM1);
#elif BLAZE_OP == 5
        SFPU_UNARY_CALL(
            DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_scaled_tanh, (is_fp32_dest_acc_en, 32), 0, VectorMode::None, BLAZE_PARAM0, BLAZE_PARAM1);
#elif BLAZE_OP == 6
        SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, semantic::calculate_logit_softcap, (32), 0, VectorMode::None, BLAZE_PARAM0, 0u /* unused */);
#elif BLAZE_OP == 7
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            calculate_silu_scaled,
            (is_fp32_dest_acc_en, true /* HAS_TAIL_SCALE */, false /* HAS_POST_SCALE */, 32),
            0,
            VectorMode::None,
            BLAZE_PARAM0,
            BLAZE_PARAM1);
#elif BLAZE_OP == 8
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            _sparse_k_filter_tile_,
            (32 /* ITERATIONS */, BSKF_BANK_MASK, BSKF_MY_BANK, BSKF_GLOBAL_BANK_SHIFT, BSKF_WITHIN_BANK_MASK, BSKF_OUT_SHIFT),
            0,
            VectorMode::None);
#elif BLAZE_OP == 9
        SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, _zero_pad_tile_, (is_fp32_dest_acc_en, BZP_VALID_ROWS, BZP_TOTAL_ROWS), 0, VectorMode::None);
#elif BLAZE_OP == 10
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            calculate_add_rsqrt,
            (APPROX_MODE, 8, is_fp32_dest_acc_en, false /* FAST_APPROX */),
            0,
            VectorMode::RC,
            BLAZE_PARAM0 /* eps bits */);
#elif BLAZE_OP == 11
        SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DstSync::SyncHalf, is_fp32_dest_acc_en, blaze_sdpa_exp_tile, 0, VectorMode::None);
#elif BLAZE_OP == 12
#if BLAZE_IMPL == 0
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            sfpu_rope_all_rows,
            (1 /* Ht */, 1 /* Wt */, 0 /* x_base */, 64 /* x_stride */, 64 /* cos_base */, 96 /* sin_base */, 16 /* cs_stride */, false /* has_scale */),
            0,
            VectorMode::RC_custom,
            0u /* scale */);
#else
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            semantic::sfpu_rope_all_rows,
            (1 /* Ht */, 1 /* Wt */, 0 /* x_base */, 64 /* x_stride */, 64 /* cos_base */, 96 /* sin_base */, 16 /* cs_stride */, false /* has_scale */),
            0,
            VectorMode::RC_custom,
            0u /* scale */);
#endif
#elif BLAZE_OP == 13
#if BLAZE_IMPL == 0
#if BLAZE_SUBOP == 0
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            _calculate_sdpa_reduce_max_row_8x32_,
            (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */, 1),
            0,
            VectorMode::RC_custom,
            0u /* src_index */,
            0u /* dst_index */,
            false /* prev_max */);
#else
            SFPU_UNARY_CALL(
                DstSync::SyncHalf,
                is_fp32_dest_acc_en,
                _calculate_sdpa_reduce_sum_row_8x32_,
                (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */),
                0,
                VectorMode::RC_custom,
                0u /* src_index */,
                0u /* dst_index */,
                false /* prev_sum */);
#endif
#elif BLAZE_IMPL == 4 // lane-FK cross-lane migration (sfpi_crosslane.h frames)
#if BLAZE_SUBOP == 0
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            semantic::crosslane::_calculate_sdpa_reduce_max_row_8x32_,
            (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */, 1),
            0,
            VectorMode::RC_custom,
            0u /* src_index */,
            0u /* dst_index */,
            false /* prev_max */);
#else
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            semantic::crosslane::_calculate_sdpa_reduce_sum_row_8x32_,
            (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */),
            0,
            VectorMode::RC_custom,
            0u /* src_index */,
            0u /* dst_index */,
            false /* prev_sum */);
#endif
#elif BLAZE_IMPL == 5 || BLAZE_IMPL == 6 || BLAZE_IMPL == 7 || BLAZE_IMPL == 8 // lane-IE uniform twins (5 pair-step / 6 walk8 / 7 half / 8 seq)
#define BLAZE_UNI_SHAPE (BLAZE_IMPL == 5 ? 0 : (BLAZE_IMPL == 6 ? 1 : (BLAZE_IMPL == 7 ? 2 : 3)))
#if BLAZE_SUBOP == 0
            SFPU_UNARY_CALL(
                DstSync::SyncHalf,
                is_fp32_dest_acc_en,
                semantic::_calculate_sdpa_reduce_max_row_8x32_uniform_,
                (DataFormat::Float16_b, 4 /* block_width */, BLAZE_UNI_SHAPE /* shape */, true /* skip_signalling */, 1),
                0,
                VectorMode::RC_custom,
                0u /* src_index */,
                0u /* dst_index */,
                false /* prev_max */);
#else
            SFPU_UNARY_CALL(
                DstSync::SyncHalf,
                is_fp32_dest_acc_en,
                semantic::_calculate_sdpa_reduce_sum_row_8x32_uniform_,
                (DataFormat::Float16_b, 4 /* block_width */, BLAZE_UNI_SHAPE /* shape */, true /* skip_signalling */),
                0,
                VectorMode::RC_custom,
                0u /* src_index */,
                0u /* dst_index */,
                false /* prev_sum */);
#endif
#elif BLAZE_IMPL == 3 // lane-FI walk variant of the lift (address-invariant blocks)
#if BLAZE_SUBOP == 0
            SFPU_UNARY_CALL(
                DstSync::SyncHalf,
                is_fp32_dest_acc_en,
                semantic::_calculate_sdpa_reduce_max_row_8x32_walk_,
                (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */, 1),
                0,
                VectorMode::RC_custom,
                0u /* src_index */,
                0u /* dst_index */,
                false /* prev_max */);
#else
            SFPU_UNARY_CALL(
                DstSync::SyncHalf,
                is_fp32_dest_acc_en,
                semantic::_calculate_sdpa_reduce_sum_row_8x32_walk_,
                (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */),
                0,
                VectorMode::RC_custom,
                0u /* src_index */,
                0u /* dst_index */,
                false /* prev_sum */);
#endif
#else // BLAZE_IMPL == 2: lane-EW typed lift
#if BLAZE_SUBOP == 0
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            semantic::_calculate_sdpa_reduce_max_row_8x32_,
            (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */, 1),
            0,
            VectorMode::RC_custom,
            0u /* src_index */,
            0u /* dst_index */,
            false /* prev_max */);
#else
        SFPU_UNARY_CALL(
            DstSync::SyncHalf,
            is_fp32_dest_acc_en,
            semantic::_calculate_sdpa_reduce_sum_row_8x32_,
            (DataFormat::Float16_b, 4 /* block_width */, true /* skip_signalling */),
            0,
            VectorMode::RC_custom,
            0u /* src_index */,
            0u /* dst_index */,
            false /* prev_sum */);
#endif
#endif // BLAZE_IMPL
#else
#error "unknown BLAZE_OP"
#endif
        }

        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t PACK_FMT = UNPACK_A_IN;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
