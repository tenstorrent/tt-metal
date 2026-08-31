// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false /* is_fp32_dest_acc_en - why true does not work? */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false /* is_fp32_dest_acc_en - why true does not work? */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

// Fresh semantic bodies: keep after sfpu_operations.h in their own include
// block (their templates consume its transitive typed helpers at
// definition-time lookup; clang-format sorts blocks independently).
#include "fresh_cpp/log1p.h"
#include "fresh_cpp_operations.h"
// Storm-contract canonical per-op semantic bodies (new bodies never land in
// the legacy aggregator above).
#include "fresh_cpp/rdiv.h"
#include "fresh_cpp/relu.h"
#include "fresh_cpp/roundingops.h"
#include "fresh_cpp/rpow.h"
#include "fresh_cpp/selu.h"
#include "fresh_cpp/sign.h"
// Storm-contract semantic bodies (one op per header, fresh_cpp/README.md).
#include "fresh_cpp/abs.h"
#include "fresh_cpp/absint32.h"
#include "fresh_cpp/add1.h"
#include "fresh_cpp/bitwisenot.h"
#include "fresh_cpp/arecipprobe.h"
#include "fresh_cpp/castfp32tofp16a.h"
#include "fresh_cpp/celu.h"
#include "fresh_cpp/comp.h"
// Byte-untouched legacy tt-llk LUT sigmoid (6-segment SFPLUTFP32 hand kernel,
// laneED sem-only audit): included ONLY as the impl-3 hand arm of
// test_sigmoid_lut_fresh_cpp — no production dispatch reaches it (corpus
// manifest class D-ABSENT).
#include "sfpu/ckernel_sfpu_sigmoid.h"
// Canonical per-op semantic bodies (storm contract: fresh_cpp/README.md).
#include "fresh_cpp/acosh_fitted.h"
#include "fresh_cpp/celu_fitted.h"
#include "fresh_cpp/digamma_fitted.h"
#include "fresh_cpp/elu_fitted.h"
#include "fresh_cpp/exp_fitted.h"
#include "fresh_cpp/expm1_fitted.h"
#include "fresh_cpp/gelu_255_licensed.h"
#include "fresh_cpp/gelu_appx_licensed.h"
#include "fresh_cpp/gelu_fitted.h"
#include "fresh_cpp/i0_fitted.h"
#include "fresh_cpp/i1_fitted.h"
#include "fresh_cpp/lgamma_fitted.h"
#include "fresh_cpp/log1p_fitted.h"
#include "fresh_cpp/log_fitted.h"
#include "fresh_cpp/mish_fitted.h"
#include "fresh_cpp/polygamma_fitted.h"
#include "fresh_cpp/rsqrt_fitted.h"
#include "fresh_cpp/selu_fitted.h"
#include "fresh_cpp/sigmoid_fitted.h"
#include "fresh_cpp/sigmoid_lut_licensed.h"
#include "fresh_cpp/softplus.h"
#include "fresh_cpp/softshrink.h"
#include "fresh_cpp/softsign.h"
#include "fresh_cpp/sqrt.h"
#include "fresh_cpp/square.h"
#include "fresh_cpp/tanh.h"
#include "fresh_cpp/tanh_fitted.h"
#include "fresh_cpp/tanhderivative-lut.h"
#include "fresh_cpp/tanhderivative_fitted.h"
#include "fresh_cpp/tanhlut_licensed.h"
#include "fresh_cpp/tanhshrink.h"
#include "fresh_cpp/threshold.h"
#include "fresh_cpp/threshold_fitted.h"
#include "fresh_cpp/trigonometry.h"
#include "fresh_cpp/unarycomp.h"
#include "fresh_cpp/unarypower.h"
#include "fresh_cpp/unaryshift.h"
#include "fresh_cpp/xielu.h"

#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

using namespace ckernel;
using namespace ckernel::sfpu;

const int iterations = 32;

// Fixed dispatch scalars for the fresh unary max/min selectors, mirroring the
// production dispatch in sfpu_operations.h (0u = 0.0f for the float ops,
// MAXMIN_SCALAR = 1000 for the integer ops) and shared with the golden
// (golden_generators.py: _UNARY_MAX_MIN_VALUE / _int_maxmin_scalar). The
// production and fresh legs must always receive identical inputs.
constexpr std::uint32_t FRESH_UNARY_MAX_MIN_FLOAT_VALUE = 0u; // 0.0f
constexpr std::uint32_t FRESH_UNARY_MAX_MIN_INT_SCALAR  = 1000u;

// Fresh semantic-C++ reciprocal.  The body names only typed Dst values and
// reciprocal arithmetic; physical LREGs, macro templates, replay ranges, and
// instruction scheduling remain compiler responsibilities.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_reciprocal_semantic()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat input = sfpi::dst_reg[0];
#ifdef ARCH_BLACKHOLE
        sfpi::vFloat result = sfpi::approx_recip(input);
        if constexpr (!APPROXIMATION_MODE)
        {
            // Cubic Newton correction in value space.  min maps the NaN error
            // produced at zero/infinity to 1.0 under SFPU min semantics, so
            // the final y+y preserves the architectural pole behavior without
            // the v_if form that currently trips rvtt_expand SSA verification.
            sfpi::vFloat error = 1.0f - input * result;
            sfpi::vFloat correction = error * error + error;
            correction = correction * error + error;
            correction = sfpi::min(correction, 1.0f);
            result = correction * result + result;
        }
#else
        // Wormhole's typed reciprocal helper implements the same semantic
        // operation with its polynomial seed; approx_recip is BH-only.
        sfpi::vFloat result = sfpu_reciprocal<APPROXIMATION_MODE>(input);
#endif
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Lane-FI envelope-attack twin of calculate_reciprocal_semantic: identical
// per-element math, TWO independent rows in flight per iteration (software
// pipelining written at source).  The impl-1 body is a 9-slot fully serial
// chain (load -> arecip -> 3 dependent MADs -> min-swap -> MAD -> store, plus
// a scheduled SFPNOP pad); the replay window it forms executes ~18.7 c/row on
// silicon while the hand kernel's SFPLOADMACRO packing runs ~14.6 c/row, and
// the KERNEL (e2e drain-inclusive) metric exposes the difference the BODY
// zone hides (weekly-e2e-weekly-20260821: recip 924 vs 792 = +16.67% LOSS).
// Interleaving two rows gives the scheduler/replay window cross-row ILP with
// no algorithm change: the mechanism certificate for a future compiler
// window-pairing pass.  LREG budget: peak 6 live vector values (<= 8).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_reciprocal_semantic_ilv2()
{
    static_assert(ITERATIONS % 2 == 0, "interleave-2 body consumes rows in pairs");
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2)
    {
        sfpi::vFloat inA = sfpi::dst_reg[0];
        sfpi::vFloat inB = sfpi::dst_reg[1];
#ifdef ARCH_BLACKHOLE
        sfpi::vFloat rA = sfpi::approx_recip(inA);
        sfpi::vFloat rB = sfpi::approx_recip(inB);
        if constexpr (!APPROXIMATION_MODE)
        {
            // Same cubic Newton correction as calculate_reciprocal_semantic,
            // element-for-element; only the row pairing differs.
            sfpi::vFloat eA = 1.0f - inA * rA;
            sfpi::vFloat eB = 1.0f - inB * rB;
            sfpi::vFloat cA = eA * eA + eA;
            sfpi::vFloat cB = eB * eB + eB;
            cA              = cA * eA + eA;
            cB              = cB * eB + eB;
            cA              = sfpi::min(cA, 1.0f);
            cB              = sfpi::min(cB, 1.0f);
            rA              = cA * rA + rA;
            rB              = cB * rB + rB;
        }
#else
        sfpi::vFloat rA = sfpu_reciprocal<APPROXIMATION_MODE>(inA);
        sfpi::vFloat rB = sfpu_reciprocal<APPROXIMATION_MODE>(inB);
#endif
        sfpi::dst_reg[0] = rA;
        sfpi::dst_reg[1] = rB;
        sfpi::dst_reg += 2;
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
// copy srca to dest
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    test_utils::call_unary_sfpu_operation_init<
        SFPU_UNARY_OPERATION,
        APPROX_MODE,
        is_fp32_dest_acc_en,
        iterations,
        FAST_MODE,
        false /* STABLE_SORT */,
        CLAMP_NEGATIVE>();

#if FRESH_CPP_IMPL == 3
    // impl 3 = explicitly selected byte-untouched hand LUT variant (laneED
    // sem-only audit).  For SfpuType::sigmoid the hand arm is the legacy
    // tt-llk 6-segment SFPLUTFP32 kernel, which no production dispatch
    // reaches; its LReg coefficient loads must run in the same init frame the
    // production init used.  (For ops without an impl-3 calculate branch,
    // impl 3 falls through to the production dispatch below — e.g. the
    // approx-mode LUT tanh, selected by APPROX_MODE rather than a branch.)
    if constexpr (SFPU_UNARY_OPERATION == SfpuType::sigmoid)
    {
        llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid>(ckernel::sfpu::_init_sigmoid_<APPROX_MODE>);
    }
#endif

    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                block_tile, formats.math, formats.math);

            // calculation of sfpu operation on dest
            // calling sfpu function from ckernel
            // this part is where parametrization of operation takes part
            {
                START_PERF_MEASURE("RECIPROCAL_BODY")
            if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exponential)
            {
                // Guard only variants that actually select this branch: run_kernel is not
                // a template, so a discarded `if constexpr` branch is still fully checked
                // and an unconditional static_assert here rejects every APPROX_MODE /
                // fp32-dest variant of this kernel (signbit, reciprocal, ...).
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::exponential || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "semantic exp selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_exp_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_sigmoid_appx_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::sigmoid_appx)
            {
                // Second semantic form: 3-range magnitude dispatch tree (the
                // LUT-eligible shape); same golden/tolerance contract as impl 1.
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sigmoid_appx_tree_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::signbit)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_signbit_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_max || SFPU_UNARY_OPERATION == SfpuType::unary_min))
            {
                constexpr bool is_max = SFPU_UNARY_OPERATION == SfpuType::unary_max;
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_unary_max_min_fresh_cpp,
                    (is_max, iterations),
                    block_tile,
                    VectorMode::None,
                    FRESH_UNARY_MAX_MIN_FLOAT_VALUE);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_max_int32 || SFPU_UNARY_OPERATION == SfpuType::unary_min_int32 ||
                                        SFPU_UNARY_OPERATION == SfpuType::unary_max_uint32 || SFPU_UNARY_OPERATION == SfpuType::unary_min_uint32))
            {
                constexpr bool is_max      = SFPU_UNARY_OPERATION == SfpuType::unary_max_int32 || SFPU_UNARY_OPERATION == SfpuType::unary_max_uint32;
                constexpr bool is_unsigned = SFPU_UNARY_OPERATION == SfpuType::unary_max_uint32 || SFPU_UNARY_OPERATION == SfpuType::unary_min_uint32;
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_unary_max_min_int_fresh_cpp,
                    (is_max, is_unsigned, iterations),
                    block_tile,
                    VectorMode::None,
                    FRESH_UNARY_MAX_MIN_INT_SCALAR);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::ceil)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_ceil_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::equal_zero)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_eqz_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            // laneED sem-only audit: the remaining five float zero-comparisons
            // (production = the all-raw-TTI calculate_comp hand kernel).
            else if constexpr (
                FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::not_equal_zero || SFPU_UNARY_OPERATION == SfpuType::less_than_zero ||
                                        SFPU_UNARY_OPERATION == SfpuType::greater_than_zero || SFPU_UNARY_OPERATION == SfpuType::less_than_equal_zero ||
                                        SFPU_UNARY_OPERATION == SfpuType::greater_than_equal_zero))
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_comp_fresh_cpp, (SFPU_UNARY_OPERATION, iterations), block_tile, VectorMode::None);
            }
            // laneED sem-only audit: semantic arm for the GeluAppx contract (the
            // production body is the 6-segment SFPLUTFP32 hand kernel
            // calculate_gelu_appx; golden = exact gelu at the registered
            // GeluAppx tolerance, which the fresh exact-gelu body meets).
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::gelu_appx)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            // laneED sem-only audit: impl 3 = the byte-untouched legacy tt-llk
            // 6-segment SFPLUTFP32 sigmoid hand kernel (corpus manifest
            // legacy__ckernel_sfpu_sigmoid, class D-ABSENT: no dispatch
            // anywhere reached it before this selector).
            else if constexpr (FRESH_CPP_IMPL == 3 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sigmoid_, (APPROX_MODE, iterations), block_tile, VectorMode::None, iterations);
            }
            // Lane GI LICENSED semantic arms (owner ratification 2026-08-24,
            // review_records/OWNER-RATIFICATION-arm-preference-lut-license.md
            // item 2): impl 4 = the accuracy-licensed sem body — matches the
            // hand LUT kernel's measured accuracy contract (equal-or-better
            // error on the row's golden domain, proven exhaustively; see the
            // provenance headers in fresh_cpp/*_licensed.h).
            else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::gelu_appx)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu_appx_licensed_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::gelu)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu_255_licensed_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sigmoid_lut_licensed_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 4 && SFPU_UNARY_OPERATION == SfpuType::tanh)
            {
                static_assert(
                    FRESH_CPP_IMPL != 4 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "licensed tanh selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh_lut_licensed_cpp, (iterations), block_tile, VectorMode::None);
            }
            // Storm-lane S1 selectors (fresh_cpp/<op>.h semantic bodies).
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::abs)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_abs_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::abs_int32)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_abs_int32_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::bitwise_not)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_bitwise_not_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::add1)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_add1_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::cast_fp32_to_fp16a)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cast_fp32_to_fp16a_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            // Lane GW SFPARECIP-mode certification probes (fresh_cpp/arecipprobe.h):
            // bare EXP (impl 5) / COND_RECIP (impl 6) against the ISA
            // functional-model golden.  Hosted on SfpuType::identity (generic
            // init, R7 LLK-pristine: no SfpuType enum extension); the python
            // side keys the golden on its own MathOperation.Approx*Probe.
            else if constexpr (FRESH_CPP_IMPL == 5 && SFPU_UNARY_OPERATION == SfpuType::identity)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_approx_exp_probe_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 6 && SFPU_UNARY_OPERATION == SfpuType::identity)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_approx_cond_recip_probe_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::celu)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_celu_fresh_cpp,
                    (is_fp32_dest_acc_en, iterations, FRESH_CELU_ALPHA_BITS, FRESH_CELU_ALPHA_RECIP_BITS),
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::clamp)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_clamp_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_CLAMP_LO, FRESH_CLAMP_HI);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardtanh)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_hardtanh_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_CLAMP_LO, FRESH_CLAMP_HI);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanh)
            {
                // The fresh tanh states the bf16 production contract (polynomial +
                // bf16 RNE store); guard only variants that select this branch.
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh tanh selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            // Lane CM fitted-kernel placeholders (tt-polynomial-fitter frontier
            // selections; provenance headers in fresh_cpp/*_fitted.h): impl 2.
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::tanh)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::tanh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted tanh selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::sigmoid || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted sigmoid selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sigmoid_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::gelu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::gelu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted gelu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::tanh_derivative)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::tanh_derivative || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted tanh-derivative selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh_derivative_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            // Lane CR fitted-kernel placeholders, wave 2 (tt-polynomial-fitter
            // frontier selections; provenance in fresh_cpp/*_fitted.h).
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::digamma)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::digamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted digamma selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_digamma_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::lgamma)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::lgamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted lgamma selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_lgamma_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::polygamma)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::polygamma || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted polygamma selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_polygamma_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::i0)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::i0 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted i0 selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i0_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::i1)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::i1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted i1 selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i1_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::mish)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::mish || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted mish selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_mish_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::log)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::log || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted log selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_log_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::log1p)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::log1p || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted log1p selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_log1p_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::exponential)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::exponential || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted exponential selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_exponential_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::rsqrt)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::rsqrt || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted rsqrt selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rsqrt_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::celu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::celu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted celu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_celu_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::elu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::elu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted elu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_elu_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::selu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::selu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted selu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_selu_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::threshold)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::threshold || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted threshold selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_threshold_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::expm1)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::expm1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted expm1 selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_expm1_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::acosh)
            {
                static_assert(
                    FRESH_CPP_IMPL != 2 || SFPU_UNARY_OPERATION != SfpuType::acosh || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fitted acosh selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_acosh_fitted_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanh_derivative_lut)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh_derivative_lut_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::silu)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_silu_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::fmod)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_fmod_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None,
                    FRESH_FMOD_DIVISOR,
                    FRESH_FMOD_DIVISOR_RECIP);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::remainder)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_remainder_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None,
                    FRESH_FMOD_DIVISOR,
                    FRESH_FMOD_DIVISOR_RECIP);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::log)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_log_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::expm1)
            {
                // The fresh expm1 states the bf16 production contract; guard only
                // variants that actually select this branch.
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::expm1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh expm1 selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_expm1_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::sqrt || SFPU_UNARY_OPERATION == SfpuType::rsqrt))
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || (SFPU_UNARY_OPERATION != SfpuType::sqrt && SFPU_UNARY_OPERATION != SfpuType::rsqrt) ||
                        (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh sqrt/rsqrt selectors support only non-approx, bf16 dest");
                constexpr bool is_reciprocal = SFPU_UNARY_OPERATION == SfpuType::rsqrt;
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sqrt_rsqrt_fresh_cpp, (is_reciprocal, iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::power)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::power || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh unary power selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_unary_power_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_POWER_EXPONENT);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::xielu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::xielu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh xielu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_xielu_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_XIELU_ALPHA, FRESH_XIELU_ALPHA);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sigmoid)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::sigmoid || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh sigmoid selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sigmoid_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::cbrt)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::cbrt || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh cbrt selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cbrt_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softplus)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::softplus || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh softplus selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_softplus_fresh_cpp,
                    (iterations),
                    block_tile,
                    VectorMode::None,
                    FRESH_SOFTPLUS_BETA,
                    FRESH_SOFTPLUS_BETA_RECIP,
                    FRESH_SOFTPLUS_THRESHOLD);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardsigmoid)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_hardsigmoid_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::gelu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::gelu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh gelu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::expm1_cw)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_expm1_cw_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::i1)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::i1 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh i1 selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i1_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::floor)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_floor_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::trunc)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_trunc_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::frac)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_frac_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::rdiv)
            {
                // The fresh rdiv states the bf16-dest reciprocal contract
                // (recip rounded to bf16 before the multiply).
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::rdiv || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh rdiv selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rdiv_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_RDIV_VALUE);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::rpow)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::rpow || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh rpow selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rpow_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::selu)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::selu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh selu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_selu_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_SELU_SCALE, FRESH_SELU_ALPHA);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::sign)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sign_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::relu_max)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_relu_max_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_RELU_MAX_THRESHOLD);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::log1p)
            {
                // The fresh log1p states the bf16 production contract; guard only
                // variants that actually select this branch.
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::log1p || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh log1p selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_log1p_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            // Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::fill)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_fill_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_FILL_VALUE);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::heaviside)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_heaviside_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_HEAVISIDE_VALUE);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardshrink)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_hardshrink_fresh_cpp, (iterations), block_tile, VectorMode::None, FRESH_HARDSHRINK_LAMBDA);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::hardmish)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_hardmish_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::elu)
            {
                // The fresh elu states the bf16 contract (exp recombination +
                // bf16 RNE store); guard only variants that select this branch.
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::elu || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh elu selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_elu_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::exp2)
            {
                static_assert(
                    FRESH_CPP_IMPL != 1 || SFPU_UNARY_OPERATION != SfpuType::exp2 || (!APPROX_MODE && !is_fp32_dest_acc_en),
                    "fresh exp2 selector supports only non-approx, bf16 dest");
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_exp2_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erf)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erf_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erfc)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfc_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::erfinv)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfinv_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::digamma)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_digamma_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            // Storm S5 (fresh_cpp/ canonical per-op bodies).
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softshrink)
            {
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_softshrink_fresh_cpp, (iterations), block_tile, VectorMode::None, 0x3f000000u /* lambda = 0.5f */);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::softsign)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softsign_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::square)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_square_fresh_cpp, (iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::tanhshrink)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanhshrink_fresh_cpp, (is_fp32_dest_acc_en, iterations), block_tile, VectorMode::None);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::threshold)
            {
                // Scalars mirror the production dispatch in sfpu_operations.h (threshold 5.0f, value 10.0f).
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_threshold_fresh_cpp, (iterations), block_tile, VectorMode::None, 5.0f, 10.0f);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && (SFPU_UNARY_OPERATION == SfpuType::unary_ge || SFPU_UNARY_OPERATION == SfpuType::unary_le))
            {
                SFPU_UNARY_CALL(
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    calculate_unary_comp_fresh_cpp,
                    (SFPU_UNARY_OPERATION == SfpuType::unary_ge, iterations),
                    block_tile,
                    VectorMode::None,
                    0x3f000000u /* value = 0.5f, the production dispatch scalar */);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::left_shift)
            {
                // Shift amount mirrors the production dispatch constant (SHIFT_AMOUNT = 3u).
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_shift_fresh_cpp, (iterations), block_tile, VectorMode::None, 3u);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::acosh)
            {
                SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_acosh_fresh_cpp, (is_fp32_dest_acc_en, iterations), block_tile, VectorMode::None);
            }
            else if constexpr (RECIPROCAL_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::reciprocal)
            {
                _llk_math_eltwise_unary_sfpu_params_(
                    calculate_reciprocal_semantic<APPROX_MODE, iterations>,
                    block_tile,
                    VectorMode::None);
            }
            else if constexpr (RECIPROCAL_IMPL == 2 && SFPU_UNARY_OPERATION == SfpuType::reciprocal)
            {
                _llk_math_eltwise_unary_sfpu_params_(calculate_reciprocal_semantic_ilv2<APPROX_MODE, iterations>, block_tile, VectorMode::None);
            }
            else
            {
                test_utils::call_unary_sfpu_operation<
                    DST_SYNC,
                    is_fp32_dest_acc_en,
                    SFPU_UNARY_OPERATION,
                    APPROX_MODE,
                    is_fp32_dest_acc_en,
                    iterations,
                    FAST_MODE,
                    false /* STABLE_SORT */,
                    CLAMP_NEGATIVE>(block_tile, formats.math);
            }
            }
        }

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                block_tile, L1_ADDRESS(params.buffer_Res[block_start * params.NUM_TILES_IN_BLOCK + block_tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
