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
#include "fresh_cpp_operations.h"

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
            else if constexpr (RECIPROCAL_IMPL == 1 && SFPU_UNARY_OPERATION == SfpuType::reciprocal)
            {
                _llk_math_eltwise_unary_sfpu_params_(
                    calculate_reciprocal_semantic<APPROX_MODE, iterations>,
                    block_tile,
                    VectorMode::None);
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
