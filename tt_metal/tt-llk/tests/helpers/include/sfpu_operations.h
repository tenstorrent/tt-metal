// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_add_top_row.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_topk.h"

// To add a new metal SFPU operation:
// 1. Include the metal header below: #include "llk_sfpu/<operation>.h"
// 2. Add the operation enum to SfpuType in llk_sfpu_types.h
// 3. Add the if constexpr branches in call_unary_sfpu_operation_init() and call_unary_sfpu_operation() below
#include "ckernel_sfpu_where.h"
#include "llk_sfpu/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_activations.h"
#include "llk_sfpu/ckernel_sfpu_add1.h"
#include "llk_sfpu/ckernel_sfpu_addcdiv.h"
#include "llk_sfpu/ckernel_sfpu_addcmul.h"
#include "llk_sfpu/ckernel_sfpu_atan2.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_binary_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_binary_comp.h"
#include "llk_sfpu/ckernel_sfpu_binary_fmod.h"
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h"
#include "llk_sfpu/ckernel_sfpu_binary_remainder.h"
#include "llk_sfpu/ckernel_sfpu_bitwise_not.h"
#include "llk_sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h"
#include "llk_sfpu/ckernel_sfpu_cbrt.h"
#include "llk_sfpu/ckernel_sfpu_digamma.h"
#include "llk_sfpu/ckernel_sfpu_div_int32.h"
#include "llk_sfpu/ckernel_sfpu_div_int32_floor.h"
#include "llk_sfpu/ckernel_sfpu_erf.h"
#include "llk_sfpu/ckernel_sfpu_erfc.h"
#include "llk_sfpu/ckernel_sfpu_expm1.h"
#include "llk_sfpu/ckernel_sfpu_fmod.h"
#include "llk_sfpu/ckernel_sfpu_gcd.h"
#include "llk_sfpu/ckernel_sfpu_hardmish.h"
#include "llk_sfpu/ckernel_sfpu_hardshrink.h"
#include "llk_sfpu/ckernel_sfpu_i1.h"
#include "llk_sfpu/ckernel_sfpu_identity.h"
#include "llk_sfpu/ckernel_sfpu_isclose.h"
#include "llk_sfpu/ckernel_sfpu_lcm.h"
#include "llk_sfpu/ckernel_sfpu_lgamma.h"
#include "llk_sfpu/ckernel_sfpu_logical_not.h"
#include "llk_sfpu/ckernel_sfpu_logsigmoid.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_mul_int32.h"
#include "llk_sfpu/ckernel_sfpu_polygamma.h"
#include "llk_sfpu/ckernel_sfpu_prelu.h"
#include "llk_sfpu/ckernel_sfpu_remainder.h"
#include "llk_sfpu/ckernel_sfpu_rpow.h"
#include "llk_sfpu/ckernel_sfpu_rsub_int32.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid_appx.h"
#include "llk_sfpu/ckernel_sfpu_sign.h"
#include "llk_sfpu/ckernel_sfpu_signbit.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_sqrt_custom.h"
#include "llk_sfpu/ckernel_sfpu_tanh_derivative.h"
#include "llk_sfpu/ckernel_sfpu_unary_comp.h"
#include "llk_sfpu/ckernel_sfpu_unary_max_min.h"
#include "llk_sfpu/ckernel_sfpu_unary_power.h"
#include "llk_sfpu/ckernel_sfpu_unary_shift.h"
#include "llk_sfpu/ckernel_sfpu_xielu.h"
// This header expects a DST_ACCUM_MODE macro; scope it to the include so it
// doesn't clash with the DST_ACCUM_MODE template param used below.
#define DST_ACCUM_MODE 0
#include "llk_sfpu/ckernel_sfpu_binop_with_unary.h"
#undef DST_ACCUM_MODE
#include "llk_sfpu/ckernel_sfpu_celu.h"
#include "llk_sfpu/ckernel_sfpu_elu.h"
#include "llk_sfpu/ckernel_sfpu_erfinv.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_exp2.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_heaviside.h"
#include "llk_sfpu/ckernel_sfpu_i0.h"
#include "llk_sfpu/ckernel_sfpu_lerp.h"
#include "llk_sfpu/ckernel_sfpu_log1p.h"
#include "llk_sfpu/ckernel_sfpu_mish.h"
#include "llk_sfpu/ckernel_sfpu_rdiv.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"
#include "llk_sfpu/ckernel_sfpu_rsqrt.h"
#include "llk_sfpu/ckernel_sfpu_selu.h"
#include "llk_sfpu/ckernel_sfpu_shift.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_sfpu/ckernel_sfpu_snake_beta.h"
#include "llk_sfpu/ckernel_sfpu_softshrink.h"
#include "llk_sfpu/ckernel_sfpu_softsign.h"
#include "llk_sfpu/ckernel_sfpu_sqrt.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_tanhshrink.h"
#include "llk_sfpu/ckernel_sfpu_trigonometry.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "sfpu/ckernel_sfpu_abs.h"
#include "sfpu/ckernel_sfpu_add_int.h"
#include "sfpu/ckernel_sfpu_clamp.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_hardtanh.h"
#include "sfpu/ckernel_sfpu_isinf_isnan.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_negative.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sub_int.h"
#include "sfpu/ckernel_sfpu_tanh_derivative.h"
#include "sfpu/ckernel_sfpu_threshold.h"

// Test-only SFPU loop/adapter wrappers (calculate_sqrt_custom, calculate_expm1_cw,
// calculate_mask_binary) used by the dispatch below.
#include "sfpu_test_helpers.h"

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::sfpu;

//
// SFPU typecast dispatch.
//
// Unlike the other unary SFPU operations (which are keyed by a single SfpuType),
// typecast selects one of ~25 `calculate_typecast_*` primitives based on the
// (input, output) DataFormat pair. The two helpers below are a faithful,
// parametrized mirror of the production compute API `typecast_tile<IN, OUT>` /
// `typecast_tile_init<IN, OUT>` (tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h),
// rewritten to take explicit template parameters instead of the ambient
// compute-kernel macros. They call the *exact same* `init_typecast_*` /
// `calculate_typecast_*` LLK primitives through the same SFPU_UNARY_* macros
// production uses. Pairs realised purely by unpacker/packer format conversion
// issue no SFPU call: they simply match no branch in the calculate helper (the
// init helper still falls through to the bare `SFPU_UNARY_INIT(typecast)`).
//
// These are reached through the shared `call_unary_sfpu_operation[_init]`
// dispatch below via `SfpuType::typecast` (with IN/OUT supplied as the trailing
// template parameters). Keep this dispatch in lockstep with typecast.h.
//
template <DataFormat IN, DataFormat OUT, bool APPROX_MODE>
void call_unary_typecast_operation_init()
{
    if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt16 && (OUT == DataFormat::UInt32 || OUT == DataFormat::Int32))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_uint32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_uint16, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_uint16, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt16 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::Int32 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt32 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_uint16, (APPROX_MODE));
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_uint8, (APPROX_MODE));
    }
    else if constexpr ((IN == DataFormat::Int32 || IN == DataFormat::UInt32 || IN == DataFormat::UInt16) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint_to_uint8, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt8 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_uint16, (APPROX_MODE));
    }
    else
    {
        // Pairs handled purely by unpacker/packer (no SFPU math) or that need
        // no per-op programming still issue the bare init so the SFPU is in a
        // defined state, exactly like the production `else` branch.
        SFPU_UNARY_INIT(typecast);
    }
}

template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, DataFormat IN, DataFormat OUT, bool APPROX_MODE, int ITERATIONS = 8>
void call_unary_typecast_operation(std::uint32_t dst_index)
{
    if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp32, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Bfp8_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Bfp8_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Bfp8_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_uint32, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Int32)
    {
        // Calls same kernel as the UInt32 case.
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_uint32, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS, DST_ACCUM_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Bfp4_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Bfp4_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Bfp4_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint8, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr ((IN == DataFormat::Int32 || IN == DataFormat::UInt32 || IN == DataFormat::UInt16) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint_to_uint8, (APPROX_MODE, ITERATIONS, (IN == DataFormat::UInt16)), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
}

/**
 * Calls only the init portion of a unary SFPU operation.
 * Must be paired with a subsequent call_unary_sfpu_operation() for the calculate step.
 * Delegates to llk_math_eltwise_unary_sfpu_init: the two-arg overload runs
 * _llk_math_eltwise_unary_sfpu_init_ then the per-operation init lambda;
 * the zero-arg overload runs only _llk_math_eltwise_unary_sfpu_init_.
 *
 * Note: `OPERATION` is a template parameter rather than a literal enumerator,
 * so the SFPU_UNARY_INIT* macros (which token-paste `::SfpuType::OP`) cannot be
 * used here — we call `llk_math_eltwise_unary_sfpu_init<OPERATION>(...)`
 * directly, which is exactly what those macros expand to.
 *
 * @tparam OPERATION The SFPU operation type to initialize
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 */
template <
    SfpuType OPERATION,
    bool APPROX_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    bool FAST_MODE          = false,
    bool STABLE_SORT        = false,
    bool CLAMP_NEGATIVE     = false,
    DataFormat TYPECAST_IN  = DataFormat::Invalid,
    DataFormat TYPECAST_OUT = DataFormat::Invalid>
void call_unary_sfpu_operation_init()
{
    if constexpr (OPERATION == SfpuType::acosh || OPERATION == SfpuType::asinh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(init_inverse_hyperbolic<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::atanh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(init_atanh<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::cosine)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(cosine_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::tan)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(tangent_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::atan)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(atan_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::sinh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sinh_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::cosh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(cosh_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(exp2_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(exp_init<APPROX_MODE, 0x3F800000 /* exp_base_scale_factor */, CLAMP_NEGATIVE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::exp_with_base)
    {
        // "exp with base b" = b^x = exp(x * ln b); implemented as the SCALE_EN path
        // of calculate_exponential (multiplies the input by a bf16 scale before exp).
        // Init is identical to exponential; the scale is applied in the calculate call.
        llk_math_eltwise_unary_sfpu_init<OPERATION>(exp_init<APPROX_MODE, 0x3F800000 /* exp_base_scale_factor */, CLAMP_NEGATIVE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(erfinv_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::erf)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(erf_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::erfc)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(erfc_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(expm1_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(cube_root_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::i1)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(i1_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::signbit)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(signbit_init);
    }
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(lgamma_stirling_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(digamma_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        // divisor = 2.0f, 1/divisor = 0.5f (fixed dispatch constants).
        llk_math_eltwise_unary_sfpu_init<OPERATION>(init_fmod<APPROX_MODE>, 0x40000000u /* 2.0f */, 0x3f000000u /* 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(init_remainder<APPROX_MODE>, 0x40000000u /* 2.0f */, 0x3f000000u /* 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sfpu_binary_pow_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::power)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sfpu_unary_pow_init);
    }
    else if constexpr (OPERATION == SfpuType::unary_max)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(unary_max_min_init<true>);
    }
    else if constexpr (OPERATION == SfpuType::unary_min)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(unary_max_min_init<false>);
    }
    else if constexpr (OPERATION == SfpuType::unary_max_int32)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(unary_max_min_int32_init<true /* IS_MAX_OP */, false /* IS_UNSIGNED */>);
    }
    else if constexpr (OPERATION == SfpuType::unary_min_int32)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(unary_max_min_int32_init<false /* IS_MAX_OP */, false /* IS_UNSIGNED */>);
    }
    else if constexpr (OPERATION == SfpuType::unary_max_uint32)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(unary_max_min_int32_init<true /* IS_MAX_OP */, true /* IS_UNSIGNED */>);
    }
    else if constexpr (OPERATION == SfpuType::unary_min_uint32)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(unary_max_min_int32_init<false /* IS_MAX_OP */, true /* IS_UNSIGNED */>);
    }
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(polygamma_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(xielu_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid_appx)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sigmoid_appx_init);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sigmoid_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::mish)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(mish_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(rdiv_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(gelu_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::gelu_appx)
    {
        // gelu_appx is the LUT branch of calculate_gelu; its init must load the
        // piecewise-linear LReg table, which only gelu_init<APPROXIMATION_MODE=true>
        // does — so force the approx init regardless of the harness APPROX_MODE.
        llk_math_eltwise_unary_sfpu_init<OPERATION>(gelu_init<true, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::gelu_derivative)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(gelu_derivative_polynomial_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(init_softsign<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::gelu_tanh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(gelu_tanh_init<is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(hardsigmoid_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_log_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::log_with_base)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(_init_log_<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::log1p)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(log1p_init<APPROX_MODE, FAST_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(recip_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(rsqrt_init<APPROX_MODE, false /* legacy_compat */>);
    }
    else if constexpr (OPERATION == SfpuType::sine)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sine_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sqrt_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(tanh_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(sfpu::tanhshrink_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::tanh_derivative_lut)
    {
        // Legacy LUT tanh': tanh_derivative_init loads the tanh piecewise-linear
        // LUT into LReg0/1/2, which _calculate_tanh_derivative_ then consumes.
        llk_math_eltwise_unary_sfpu_init<OPERATION>(tanh_derivative_init<APPROX_MODE>);
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        // Typecast selects its concrete init from the (IN, OUT) format pair.
        call_unary_typecast_operation_init<TYPECAST_IN, TYPECAST_OUT, APPROX_MODE>();
    }
    else
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>();
    }
}

/**
 * Calls only the calculate portion of a unary SFPU operation.
 * Must be preceded by a call to call_unary_sfpu_operation_init() for the same operation.
 * Delegates to SFPU_UNARY_CALL from llk_math_eltwise_unary_sfpu_macros.h,
 * which runs the ckernel::_sfpu_check_<DST_SYNC_MODE, DST_ACCUM_MODE>
 * dst-bound LLK_ASSERT and then dispatches directly to
 * _llk_math_eltwise_unary_sfpu_params_. Face-looping versus single-call
 * behavior is selected by the explicit vector_mode parameter;
 * the default/non-face mode preserves the existing single-call full-tile behavior.
 *
 * DST_SYNC_MODE and DST_ACCUM_MODE are the first two template parameters, mirroring
 * the convention of the underlying SFPU macros and helpers (SFPU_UNARY_CALL,
 * _sfpu_check_, etc.) where the dst-sync/accum pair always leads. They are
 * forwarded to _sfpu_check_ so the dst-bound LLK_ASSERT is computed against
 * the kernel's actual sync/accumulation mode.
 *
 * @tparam DST_SYNC_MODE Kernel's DstSync mode (drives the dst-bound assert)
 * @tparam DST_ACCUM_MODE Kernel's dest-accumulation flag (drives the dst-bound assert)
 * @tparam OPERATION The SFPU operation type to execute
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 * @param dst_index Destination tile index in the destination register
 * @param math_format Optional math format for operations that need format-specific behavior
 */
template <
    DstSync DST_SYNC_MODE,
    bool DST_ACCUM_MODE,
    SfpuType OPERATION,
    bool APPROX_MODE,
    bool is_fp32_dest_acc_en,
    int ITERATIONS,
    bool FAST_MODE          = false,
    bool STABLE_SORT        = false,
    bool CLAMP_NEGATIVE     = false,
    DataFormat TYPECAST_IN  = DataFormat::Invalid,
    DataFormat TYPECAST_OUT = DataFormat::Invalid>
void call_unary_sfpu_operation(std::uint32_t dst_index, std::uint32_t math_format = 0, float fill_const_value = 5.0f, VectorMode vector_mode = VectorMode::None)
{
    if constexpr (OPERATION == SfpuType::abs)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_abs_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::abs_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_abs_int32, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::add1)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_add1, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::cast_fp32_to_fp16a)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, cast_fp32_to_fp16a, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::acosh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_acosh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::asinh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_asinh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::atanh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_atanh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_celu,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x3f800000u /* alpha = 1.0f */,
            0x3f800000u /* 1/alpha = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::cosine)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_cosine, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::tan)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tangent, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::atan)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_atan, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::asin)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_asin, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::acos)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_acos, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sinh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sinh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::cosh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_cosh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_elu,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x3f800000u /* alpha = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_exp2, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    // VectorMode::RC: params drives 4 face iterations with 2×SETRWC between each —
    // the lambda processes 8 rows per face, giving 32 total.
    else if constexpr (OPERATION == SfpuType::exponential && APPROX_MODE && CLAMP_NEGATIVE)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_exponential,
            (APPROX_MODE, is_fp32_dest_acc_en, false /* scale_en */, 8, CLAMP_NEGATIVE),
            dst_index,
            VectorMode::RC,
            p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    // Single call (else branch): calculate_exponential handles 8 or 32 iterations internally.
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_exponential,
            (APPROX_MODE, is_fp32_dest_acc_en, false /* scale_en */, ITERATIONS, CLAMP_NEGATIVE),
            dst_index,
            vector_mode,
            p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    // exp_with_base = b^x = exp(x * ln b): the only op that drives calculate_exponential
    // with SCALE_EN=true. The bf16 scale 0x3F00 == 0.5 selects base b = e^0.5, so the
    // golden is exp(0.5*x); 0.5 is exact in bf16 so no scale-rounding error is added.
    else if constexpr (OPERATION == SfpuType::exp_with_base)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_exponential,
            (APPROX_MODE, is_fp32_dest_acc_en, true /* scale_en */, ITERATIONS, CLAMP_NEGATIVE),
            dst_index,
            vector_mode,
            0x3F00u /* bf16(0.5) exp base scale */);
    }
    else if constexpr (OPERATION == SfpuType::fill)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _calculate_fill_int_,
                (APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS),
                dst_index,
                vector_mode,
                static_cast<std::uint32_t>(fill_const_value));
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _calculate_fill_int_,
                (APPROX_MODE, ckernel::InstrModLoadStore::LO16, ITERATIONS),
                dst_index,
                vector_mode,
                static_cast<std::uint32_t>(fill_const_value));
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _calculate_fill_int_,
                (APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS),
                dst_index,
                vector_mode,
                static_cast<std::uint32_t>(fill_const_value));
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_fill_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, fill_const_value);
        }
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::gelu_appx)
    {
        // Directly exercise the LUT approximation kernel (the APPROXIMATION_MODE=true
        // branch of calculate_gelu). Requires the LReg table loaded by gelu_init<true>.
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu_appx, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::gelu_derivative)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu_derivative_polynomial, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::gelu_tanh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu_tanh, (is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_activation, (APPROX_MODE, ckernel::ActivationType::Hardsigmoid, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_log_,
            (APPROX_MODE, false, ITERATIONS),
            dst_index,
            vector_mode,
            ITERATIONS,
            0u /* log_base_scale_factor */);
    }
    else if constexpr (OPERATION == SfpuType::log_with_base)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_log_,
            (APPROX_MODE, true, ITERATIONS),
            dst_index,
            vector_mode,
            ITERATIONS,
            0x3DC5u /* 1/ln(2) in fp16a -> log2(x) */);
    }
    else if constexpr (OPERATION == SfpuType::log1p)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log1p, (APPROX_MODE, FAST_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::negative)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_negative_int_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_negative_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE, false /* legacy_compat */),
            dst_index,
            vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_silu_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanhshrink, (is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::floor)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_floor_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::round)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_round_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0 /* decimals */);
    }
    else if constexpr (OPERATION == SfpuType::ceil)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_ceil_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::trunc)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_trunc_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::frac)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_frac_, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sine)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sine, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_square, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::signbit)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_signbit_int32, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_signbit, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::threshold)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_threshold_,
            (APPROX_MODE, ITERATIONS, float),
            dst_index,
            vector_mode,
            5.0f /* threshold_value */,
            10.0f /* replacement_value */);
    }
    else if constexpr (OPERATION == SfpuType::topk_local_sort)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_topk_phases_steps,
            (APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT),
            dst_index,
            vector_mode,
            0 /* idir */,
            5 /* i_end_phase */,
            0 /* i_start_phase */,
            10 /* i_end_step */,
            0 /* i_start_step */);
    }
    else if constexpr (OPERATION == SfpuType::topk_merge)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_topk_merge,
            (APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT),
            dst_index,
            vector_mode,
            5 /* m_iter */,
            10 /* k */);
    }
    else if constexpr (OPERATION == SfpuType::topk_rebuild)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _bitonic_topk_rebuild,
            (APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT),
            dst_index,
            vector_mode,
            false /* idir */,
            5 /* m_iter */,
            10 /* k */,
            3 /* logk */,
            0 /* skip_second */);
    }
    else if constexpr (OPERATION == SfpuType::relu_max)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, _relu_max_, (sfpi::vFloat, APPROX_MODE, ITERATIONS, float), dst_index, vector_mode, 5.0f /* threshold */);
    }
    else if constexpr (OPERATION == SfpuType::relu_min)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, _relu_min_, (sfpi::vInt, APPROX_MODE, ITERATIONS, std::uint32_t), dst_index, vector_mode, 5u /* threshold */);
        }
        else
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, _relu_min_, (sfpi::vFloat, APPROX_MODE, ITERATIONS, float), dst_index, vector_mode, 5.0f /* threshold */);
        }
    }
    else if constexpr (OPERATION == SfpuType::lrelu)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_lrelu_, (APPROX_MODE), dst_index, vector_mode, ITERATIONS, 0x3dcccccdu /* slope = 0.1f */);
    }
    else if constexpr (OPERATION == SfpuType::add_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_add_int32, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 5u /* scalar */);
    }
    else if constexpr (OPERATION == SfpuType::sub_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sub_int32, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 5u /* scalar */);
    }
    else if constexpr (OPERATION == SfpuType::bitwise_not)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_bitwise_not, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        // logical_not(x) = (x == 0) ? 1 : 0. Select the layout from the runtime input format,
        if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_logical_not, (APPROX_MODE, ckernel::InstrModLoadStore::LO16, ITERATIONS), dst_index, vector_mode);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::Int32) || math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_logical_not, (APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS), dst_index, vector_mode);
        }
        else
        {
            SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_logical_not, (APPROX_MODE, ckernel::InstrModLoadStore::DEFAULT, ITERATIONS), dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuType::heaviside)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_heaviside, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softshrink, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* lambda = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softsign, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::mish)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mish, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::selu)
    {
        // selu constants passed as fp32 bit patterns: scale ~= 1.0507, alpha ~= 1.6733.
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_selu,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x3f867d5fu /* scale */,
            0x3fd62d7du /* alpha */);
    }
    else if constexpr (OPERATION == SfpuType::i0)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_i0, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rdiv,
            (APPROX_MODE, is_fp32_dest_acc_en, ckernel::RoundingMode::None, ITERATIONS),
            dst_index,
            vector_mode,
            0x40000000u /* value = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuType::clamp)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_clamp_,
            (APPROX_MODE, ITERATIONS),
            dst_index,
            vector_mode,
            ITERATIONS,
            0xBC00u /* min = -1.0 (fp16) */,
            0x3C00u /* max =  1.0 (fp16) */,
            0x0000u /* offset = 0 (bf16) */);
    }
    else if constexpr (OPERATION == SfpuType::hardtanh)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_hardtanh_,
            (APPROX_MODE, ITERATIONS),
            dst_index,
            vector_mode,
            ITERATIONS,
            0x3F80u /* p0 = -min = 1.0 (bf16) */,
            0xC000u /* p1 = -(max-min) = -2.0 (bf16) */,
            0x3F80u /* p2 = max = 1.0 (bf16) */);
    }
    else if constexpr (
        OPERATION == SfpuType::equal_zero || OPERATION == SfpuType::not_equal_zero || OPERATION == SfpuType::less_than_zero ||
        OPERATION == SfpuType::greater_than_zero || OPERATION == SfpuType::less_than_equal_zero || OPERATION == SfpuType::greater_than_equal_zero)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_zero_comp_, (APPROX_MODE, OPERATION, ITERATIONS), dst_index, vector_mode, 0u /* exponent_size_8 */);
    }
    else if constexpr (
        OPERATION == SfpuType::isinf || OPERATION == SfpuType::isposinf || OPERATION == SfpuType::isneginf || OPERATION == SfpuType::isnan ||
        OPERATION == SfpuType::isfinite)
    {
        // Predicate ops: write 1.0f where the (isinf/isposinf/isneginf/isnan/isfinite)
        // test holds, else 0.0f. The concrete predicate is selected by OPERATION.
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_sfpu_isinf_isnan_, (OPERATION, APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erfinv, (APPROX_MODE), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::erf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erf, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::erfc)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erfc, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_expm1, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_cube_root, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::i1)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_i1, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sign)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sign, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0u /* exponent_size_8 */);
    }
    else if constexpr (OPERATION == SfpuType::tanh_derivative)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh_derivative_sech2, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::tanh_derivative_lut)
    {
        // Legacy tt-llk primitive: computes 1 - tanh(x)^2 with tanh from the LUT
        // (WITH_PRECOMPUTED_TANH = 0). Distinct from the accurate sech2 variant above.
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_tanh_derivative_, (APPROX_MODE, 0, ITERATIONS), dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, hardmish, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_lgamma_stirling, (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_digamma, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::identity)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_identity, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_prelu, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3e800000u /* slope = 0.25f */);
    }
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rpow,
            (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en),
            dst_index,
            vector_mode,
            0x40000000u /* base = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuType::power)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_unary_power,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x40000000u /* exponent = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        // calculate_fmod takes (value, recip); the bodies read vConstFloatPrgm0/1 set by init,
        // so the runtime args are inert but the signature still requires them. Mirror init.
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_fmod,
            (APPROX_MODE, ITERATIONS),
            dst_index,
            vector_mode,
            0x40000000u /* value = 2.0f */,
            0x3f000000u /* recip = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_remainder,
            (APPROX_MODE, ITERATIONS),
            dst_index,
            vector_mode,
            0x40000000u /* value = 2.0f */,
            0x3f000000u /* recip = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_gt)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_gt, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_ne)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_ne, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_eq)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_eq, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_lt)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_lt, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_ge)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_ge, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_le)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_le, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_max)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min, (true, APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0u /* value = 0.0f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_min)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min, (false, APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0u /* value = 0.0f */);
    }
    // Integer unary max/min against a fixed scalar (1000). IS_UNSIGNED selects the
    // uint32 vs int32 SFPSWAP handling. The golden compares against the same 1000.
    else if constexpr (OPERATION == SfpuType::unary_max_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min_int32, (true, false, APPROX_MODE, ITERATIONS), dst_index, vector_mode, 1000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_min_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min_int32, (false, false, APPROX_MODE, ITERATIONS), dst_index, vector_mode, 1000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_max_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min_int32, (true, true, APPROX_MODE, ITERATIONS), dst_index, vector_mode, 1000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_min_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_unary_max_min_int32, (false, true, APPROX_MODE, ITERATIONS), dst_index, vector_mode, 1000u);
    }
    // Unary shift by a fixed immediate (3 bits). Integer-only kernels; the DATA_FORMAT
    // template is chosen from the runtime math_format. The golden shifts by the same 3.
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_left_shift, (APPROX_MODE, DataFormat::UInt16, ITERATIONS), dst_index, vector_mode, 3u);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_left_shift, (APPROX_MODE, DataFormat::UInt32, ITERATIONS), dst_index, vector_mode, 3u);
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_left_shift, (APPROX_MODE, DataFormat::Int32, ITERATIONS), dst_index, vector_mode, 3u);
        }
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_right_shift, (APPROX_MODE, DataFormat::UInt16, ITERATIONS), dst_index, vector_mode, 3u);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_right_shift, (APPROX_MODE, DataFormat::UInt32, ITERATIONS), dst_index, vector_mode, 3u);
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_right_shift, (APPROX_MODE, DataFormat::Int32, ITERATIONS), dst_index, vector_mode, 3u);
        }
    }
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        // order n = 1 (trigamma); scale = (-1)^(n+1) * n! = 1.0f.
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_polygamma,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x3f800000u /* n = 1.0f */,
            0x3f800000u /* scale = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_xielu,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x3f800000u /* alpha_p = 1.0f */,
            0x3f800000u /* alpha_n = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::hardshrink)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_hardshrink, (APPROX_MODE, ITERATIONS), dst_index, vector_mode, 0x3f000000u /* lambda = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::softplus)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_softplus,
            (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            0x3f800000u /* beta = 1.0f */,
            0x3f800000u /* 1/beta = 1.0f */,
            0x41a00000u /* threshold = 20.0f */);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid_appx)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid_appx, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::sqrt_custom)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt_custom, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt_compat)
    {
        // Legacy-compat rsqrt: reciprocal-root method (legacy_compat = true routes
        // calculate_rsqrt to _calculate_rsqrt_compat_). Distinct from SfpuType::rsqrt,
        // which exercises the accurate legacy_compat = false path.
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsqrt,
            (APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE, true /* legacy_compat */),
            dst_index,
            vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::expm1_cw)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_expm1_cw, (APPROX_MODE, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        call_unary_typecast_operation<DST_SYNC_MODE, DST_ACCUM_MODE, TYPECAST_IN, TYPECAST_OUT, APPROX_MODE, ITERATIONS>(dst_index);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

/**
 * Calls only the init portion of a binary SFPU operation.
 * Must be paired with a subsequent call_binary_sfpu_operation() for the calculate step.
 * Uses SFPU_BINARY_INIT* macros from llk_math_eltwise_binary_sfpu_macros.h, which
 * wrap `ckernel::llk_math_eltwise_binary_sfpu_init<::SfpuType::OP>(...)`.
 *
 * DST_ACCUM_MODE (is_fp32_dest_acc_en) must match the value passed to the paired
 * call_binary_sfpu_operation(): ops such as atan2 load an fp32- vs bf16-specific
 * reciprocal polynomial at init time that the calculate step then relies on.
 */
template <bool APPROXIMATION_MODE, bool DST_ACCUM_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation_init()
{
    if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV || BINOP == BinaryOp::RSUB ||
        BINOP == BinaryOp::XLOGY)
    {
        // BinaryOps without a dedicated SfpuType use the baseline binary addrmod setup.
        SFPU_BINARY_INIT_FN(add1, sfpu_binary_init, (APPROXIMATION_MODE, BINOP));
    }
    else if constexpr (BINOP == BinaryOp::POW)
    {
        SFPU_BINARY_INIT_FN(power, sfpu_binary_init, (APPROXIMATION_MODE, BINOP));
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(add1, init_add_top_row);
    }
    else if constexpr (BINOP == BinaryOp::RSHFT || BINOP == BinaryOp::LOGICAL_RSHFT)
    {
        SFPU_BINARY_INIT(right_shift);
    }
    else if constexpr (BINOP == BinaryOp::LSHFT)
    {
        SFPU_BINARY_INIT(left_shift);
    }
    else if constexpr (
        BINOP == BinaryOp::LT || BINOP == BinaryOp::GT || BINOP == BinaryOp::LE || BINOP == BinaryOp::GE || BINOP == BinaryOp::EQ || BINOP == BinaryOp::NE)
    {
        SFPU_BINARY_INIT(lt);
    }
    else if constexpr (BINOP == BinaryOp::MAX)
    {
        // binary_max_min uses SFPLOADMACRO templates programmed by its init.
        // The SfpuType must be max/min (not a generic placeholder): the LLK init
        // selects a max/min-specific address-mod setup (ADDR_MOD_6 with dest incr=2)
        // that the kernel's load/swap/store sequence depends on. Using add1 skips
        // that setup and leaves the output tile holding in0. Mirrors binary_max_tile_init.
        SFPU_BINARY_INIT_FN(max, binary_max_min_init, (true /* IS_MAX */));
    }
    else if constexpr (BINOP == BinaryOp::MIN)
    {
        // See MAX above: the min SfpuType drives the required address-mod setup.
        SFPU_BINARY_INIT_FN(min, binary_max_min_init, (false /* IS_MAX */));
    }
    else if constexpr (BINOP == BinaryOp::FMOD)
    {
        // fmod uses the reciprocal path; init loads the reciprocal polynomial.
        SFPU_BINARY_INIT_FN(add1, fmod_binary_init, (APPROXIMATION_MODE));
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER)
    {
        // is_fp32_dest_acc_en only selects the (inert) legacy_compat=false branch of
        // recip_init, so pass false here; both paths just load the reciprocal polynomial.
        SFPU_BINARY_INIT_FN(add1, remainder_binary_init, (APPROXIMATION_MODE, false));
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32)
    {
        // Truncating int32 division writes an int32 quotient (calculate_div_int32_trunc),
        // so it needs the reciprocal-polynomial constants from div_trunc_init (shared with
        // DIV_INT32_FLOOR) rather than div_init's sfpu_reciprocal_init.
        SFPU_BINARY_INIT_FN(add1, div_trunc_init, (APPROXIMATION_MODE));
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32_FLOOR)
    {
        SFPU_BINARY_INIT_FN(add1, div_floor_init, (APPROXIMATION_MODE));
    }
    else if constexpr (BINOP == BinaryOp::GCD)
    {
        // gcd_init records the per-iteration REPLAY buffer used by the binary-GCD loop.
        SFPU_BINARY_INIT_FN_NO_ARGS(add1, sfpu::calculate_sfpu_gcd_init);
    }
    else if constexpr (BINOP == BinaryOp::LCM)
    {
        // lcm_init records the binary-GCD REPLAY buffer (shared with gcd) and loads
        // the reciprocal-polynomial constants used to divide by gcd(a, b).
        SFPU_BINARY_INIT_FN_NO_ARGS(lcm, sfpu::calculate_sfpu_lcm_init);
    }
    else if constexpr (BINOP == BinaryOp::ATAN2)
    {
        // atan2 has no dedicated SfpuType, so use the baseline add1 addrmod. Its init
        // just loads the reciprocal polynomial; the is_fp32_dest_acc_en variant selects
        // the fp32 vs bf16 reciprocal (matching the minimax branch in the kernel), so it
        // must match DST_ACCUM_MODE used by the paired call_binary_sfpu_operation().
        SFPU_BINARY_INIT_FN(add1, sfpu::calculate_sfpu_atan2_init, (APPROXIMATION_MODE, DST_ACCUM_MODE));
    }
    else if constexpr (BINOP == BinaryOp::MUL_INT32)
    {
        // mul_int32 is on the LLK init's ADDR_MOD_6 dest+=2 allow-list, so drive its
        // addrmod through SfpuType::mul_int32; the init loads the 11-bit chunk masks
        // and the 2**23 fp32-to-int bias constants into vConst registers.
        SFPU_BINARY_INIT_FN(mul_int32, sfpu::mul_int32_init, (APPROXIMATION_MODE));
    }
    // Integer relational eq/ne: no extra init function, just the eq_int/ne_int addrmod.
    else if constexpr (BINOP == BinaryOp::EQ_INT)
    {
        SFPU_BINARY_INIT(eq_int);
    }
    else if constexpr (BINOP == BinaryOp::NE_INT)
    {
        SFPU_BINARY_INIT(ne_int);
    }
    // Integer max/min: binary_max_min_int32_init programs the SFPLOADMACRO templates
    // and is keyed on the matching {max,min}_{int32,uint32} SfpuType addrmod allow-list.
    else if constexpr (BINOP == BinaryOp::MAX_INT32)
    {
        SFPU_BINARY_INIT_FN(max_int32, binary_max_min_int32_init, (true /* IS_MAX */, false /* IS_UNSIGNED */));
    }
    else if constexpr (BINOP == BinaryOp::MIN_INT32)
    {
        SFPU_BINARY_INIT_FN(min_int32, binary_max_min_int32_init, (false /* IS_MAX */, false /* IS_UNSIGNED */));
    }
    else if constexpr (BINOP == BinaryOp::MAX_UINT32)
    {
        SFPU_BINARY_INIT_FN(max_uint32, binary_max_min_int32_init, (true /* IS_MAX */, true /* IS_UNSIGNED */));
    }
    else if constexpr (BINOP == BinaryOp::MIN_UINT32)
    {
        SFPU_BINARY_INIT_FN(min_uint32, binary_max_min_int32_init, (false /* IS_MAX */, true /* IS_UNSIGNED */));
    }
    // Integer remainder / fmod: init loads the reciprocal-polynomial constants used by
    // the internal 32-bit remainder kernel (shared uint path).
    else if constexpr (BINOP == BinaryOp::REMAINDER_INT32)
    {
        SFPU_BINARY_INIT_FN(remainder_int32, remainder_int32_init, (APPROXIMATION_MODE));
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_UINT32)
    {
        SFPU_BINARY_INIT_FN(remainder_uint32, remainder_uint32_init, (APPROXIMATION_MODE));
    }
    else if constexpr (BINOP == BinaryOp::FMOD_INT32)
    {
        SFPU_BINARY_INIT_FN(fmod_int32, fmod_int32_init, (APPROXIMATION_MODE));
    }
    else
    {
        // BinaryOps without a dedicated SfpuType use the baseline binary addrmod setup.
        // BITWISE_AND/OR/XOR, RSUB_INT32, MASK, ISCLOSE and LOGSIGMOID land here: those
        // kernels need no per-op init beyond the standard binary addrmod configuration
        // (logsigmoid_init is a no-op).
        SFPU_BINARY_INIT(add1);
    }
}

template <BinaryOp BINOP>
constexpr SfpuType get_binary_comp_sfpu_type()
{
    if constexpr (BINOP == BinaryOp::LT)
    {
        return SfpuType::lt;
    }
    else if constexpr (BINOP == BinaryOp::GT)
    {
        return SfpuType::gt;
    }
    else if constexpr (BINOP == BinaryOp::LE)
    {
        return SfpuType::le;
    }
    else if constexpr (BINOP == BinaryOp::GE)
    {
        return SfpuType::ge;
    }
    else if constexpr (BINOP == BinaryOp::EQ)
    {
        return SfpuType::eq;
    }
    else
    {
        return SfpuType::ne;
    }
}

/**
 * Calls only the calculate portion of a binary SFPU operation.
 * Must be preceded by a call to call_binary_sfpu_operation_init() for the same operation.
 * Uses SFPU_BINARY_CALL from llk_math_eltwise_binary_sfpu_macros.h, which
 * runs the ckernel::_sfpu_binary_check_<DST_SYNC_MODE, DST_ACCUM_MODE>
 * dst-bound LLK_ASSERTs and then dispatches directly to
 * _llk_math_eltwise_binary_sfpu_params_. The callable receives
 * (dst_index_in0, dst_index_in1, dst_index_out) forwarded from the params
 * wrapper.
 *
 * DST_SYNC_MODE and DST_ACCUM_MODE are the first two template parameters (matching
 * the SFPU_BINARY_CALL / _sfpu_binary_check_ convention) so the dst-bound
 * LLK_ASSERTs run against the kernel's actual sync/accumulation mode.
 */
template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation(
    const std::uint32_t dst_index_in0 = 0,
    const std::uint32_t dst_index_in1 = 1,
    const std::uint32_t dst_index_out = 0,
    ckernel::VectorMode vector_mode   = ckernel::VectorMode::RC)
{
    // NOTE: The functions invoked via SFPU_BINARY_CALL below run inside
    // _llk_math_eltwise_binary_sfpu_params_, which already loops over 4 faces
    // (for VectorMode::RC) and emits 2x TTI_SETRWC cr_d 8 between calls to
    // advance the dst-write counter. The per-call inner ITERATIONS must
    // therefore be 8 (one face's worth of SFPU rows), not 32 (a full tile),
    // matching how every production llk_math_eltwise_binary_sfpu_* wrapper
    // dispatches into _calculate_sfpu_binary_ / _calculate_*_shift_.
    static_assert(ITERATIONS == 8 || ITERATIONS == 32, "Binary SFPU tests support legacy 8/32 iteration values; execution uses 8 rows per face.");
    constexpr int PER_FACE_ITERATIONS = 8;
    if constexpr (BINOP == BinaryOp::DIV)
    {
        // Route DIV to the dedicated production kernel (calculate_sfpu_binary_div),
        // matching what div_binary_tile() dispatches. The generic calculate_sfpu_binary
        // DIV path is a legacy variant that production never uses, so isolating the real
        // kernel here lets the perf/functional harness measure and guard it directly.
        // is_fp32_dest_acc_en = DST_ACCUM_MODE selects the fp32 residual + bf16 rounding.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_div,
            (APPROXIMATION_MODE, BINOP, PER_FACE_ITERATIONS, DST_ACCUM_MODE),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::RSUB || BINOP == BinaryOp::XLOGY ||
        BINOP == BinaryOp::POW)
    {
        if constexpr (BINOP == BinaryOp::ADD && MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32))
        {
            SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _add_int_,
                (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32, true /* SIGN_MAGNITUDE_FORMAT */),
                dst_index_in0,
                dst_index_in1,
                dst_index_out,
                vector_mode);
        }
        else if constexpr (BINOP == BinaryOp::SUB && MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32))
        {
            // Int32 SUB must use the integer path (_sub_int_); otherwise it would
            // fall through to calculate_sfpu_binary and subtract the raw integer
            // bit-patterns as floats. Mirrors the Int32 ADD path above.
            SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                _sub_int_,
                (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32, true /* SIGN_MAGNITUDE_FORMAT */),
                dst_index_in0,
                dst_index_in1,
                dst_index_out,
                vector_mode);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sfpu_binary,
                (APPROXIMATION_MODE, BINOP, PER_FACE_ITERATIONS),
                dst_index_in0,
                dst_index_in1,
                dst_index_out,
                vector_mode);
        }
    }
    else if constexpr (BINOP == BinaryOp::RSHFT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_right_shift,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32_2S_COMP, false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LSHFT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_left_shift,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32_2S_COMP, false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LOGICAL_RSHFT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_logical_right_shift,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32_2S_COMP, false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        // Use actual format when compiling for ADD_TOP_ROW tests, otherwise use Float32 as safe default for static assert
        constexpr DataFormat add_top_row_format = (BINOP == BinaryOp::ADD_TOP_ROW) ? static_cast<DataFormat>(MATH_FORMAT) : DataFormat::Float32;
        // Force VectorMode::RC_custom so the params wrapper drives all four faces (4 x 8 = 32 rows) of the tile.
        //  _llk_math_eltwise_binary_sfpu_params_ takes its single-call branch
        //  and does not emit the per-face TTI_SETRWC
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_add_top_row,
            (add_top_row_format),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            ckernel::VectorMode::RC_custom);
    }
    else if constexpr (
        BINOP == BinaryOp::LT || BINOP == BinaryOp::GT || BINOP == BinaryOp::LE || BINOP == BinaryOp::GE || BINOP == BinaryOp::EQ || BINOP == BinaryOp::NE)
    {
        constexpr SfpuType comp_type = get_binary_comp_sfpu_type<BINOP>();
        if constexpr (MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32))
        {
            SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_binary_comp_int32,
                (APPROXIMATION_MODE, PER_FACE_ITERATIONS, comp_type),
                dst_index_in0,
                dst_index_in1,
                dst_index_out,
                vector_mode);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_binary_comp_fp32,
                (APPROXIMATION_MODE, PER_FACE_ITERATIONS, comp_type),
                dst_index_in0,
                dst_index_in1,
                dst_index_out,
                vector_mode);
        }
    }
    else if constexpr (BINOP == BinaryOp::MAX || BINOP == BinaryOp::MIN)
    {
        // float elementwise max/min (SFPSWAP min/max). Operands read from two dst tiles.
        constexpr bool IS_MAX = (BINOP == BinaryOp::MAX);
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binary_max_min, (IS_MAX, PER_FACE_ITERATIONS), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::FMOD)
    {
        // float fmod (result sign follows dividend a); DST_ACCUM_MODE selects fp32 vs bf16 store.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_fmod,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, DST_ACCUM_MODE),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER)
    {
        // float remainder (result sign follows divisor b); DST_ACCUM_MODE selects fp32 vs bf16 store.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_remainder,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, DST_ACCUM_MODE),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::BITWISE_AND || BINOP == BinaryOp::BITWISE_OR || BINOP == BinaryOp::BITWISE_XOR)
    {
        // int32 bitwise AND/OR/XOR (raw two's-complement bit patterns in dest).
        constexpr BinaryBitwiseOp BW = (BINOP == BinaryOp::BITWISE_AND)  ? BinaryBitwiseOp::AND
                                       : (BINOP == BinaryOp::BITWISE_OR) ? BinaryBitwiseOp::OR
                                                                         : BinaryBitwiseOp::XOR;
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary_bitwise,
            (APPROXIMATION_MODE, BW, ckernel::InstrModLoadStore::INT32, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32)
    {
        // int32 truncating division (rounds toward zero). calculate_div_int32_trunc writes a
        // true int32 quotient; the legacy calculate_div_int32 stored an fp32 result, which the
        // Int32 pack path reinterpreted as garbage bit patterns.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_div_int32_trunc,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32_FLOOR)
    {
        // int32 floor division (rounds toward -inf).
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_div_int32_floor,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::GCD)
    {
        // int32 gcd via the binary-GCD REPLAY loop recorded in gcd_init.
        SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_gcd, (PER_FACE_ITERATIONS), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LCM)
    {
        // int32 lcm = a/gcd(a,b) * b (binary-GCD + reciprocal); operands assumed < 2^15.
        SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_lcm, (PER_FACE_ITERATIONS), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::RSUB_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_rsub_int,
            (APPROXIMATION_MODE, ckernel::InstrModLoadStore::INT32, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::MASK)
    {
        // float mask: out = (mask != 0) ? data : 0, with data at in0 and mask at in1.
        // Driven through the test-only adapter since calculate_mask uses fixed dst
        // offsets rather than the forwarded indices.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_mask_binary,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ATAN2)
    {
        // atan2(y, x): in0 = y, in1 = x (calculate_sfpu_atan2 forwards them as
        // _sfpu_atan2_(in0, in1)). DST_ACCUM_MODE is is_fp32_dest_acc_en and selects
        // the higher-order fp32 minimax polynomial (vs the bf16 one) plus the final
        // convert-to-bf16 rounding, so it must match the init's variant.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_atan2,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, DST_ACCUM_MODE),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::MUL_INT32)
    {
        // int32 multiply: out = in0 * in1 (low 32 bits). The kernel loads/stores via
        // plain INT32 (two's-complement dest bits), so the sign-magnitude packer only
        // round-trips non-negative results; the test keeps operands positive with a
        // product < 2^31 (see test_sfpu_binary_mul_int32).
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, mul_int32, (APPROXIMATION_MODE, PER_FACE_ITERATIONS), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ISCLOSE)
    {
        // isclose: out = (|a - b| <= atol + rtol * |b|) ? 1 : 0, with a=in0, b=in1.
        // rtol/atol are passed as fp32 bit patterns via the params wrapper's runtime-arg
        // forwarding. Fixed to torch's defaults rtol=1e-5 (0x3727c5ac), atol=1e-8
        // (0x322bcc77); EQUAL_NAN=false, so any NaN operand yields 0. The test uses
        // large-margin stimuli so the exact tolerance (and fp32-vs-bf16 rounding of the
        // tol term) never flips the pass/fail decision.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_isclose,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, /*EQUAL_NAN=*/false),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode,
            /*rtol_bits=*/0x3727c5acu,
            /*atol_bits=*/0x322bcc77u);
    }
    else if constexpr (BINOP == BinaryOp::LOGSIGMOID)
    {
        // logsigmoid(x) = -softplus(-x), with x = in0 and exp(-x) = in1 (the compute
        // kernel is expected to supply exp(-x) as the second operand; the test bakes
        // it into the paired stimuli). No dedicated init (baseline add1 addrmod).
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_logsigmoid,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    // Integer relational eq/ne: XOR-based exact compare over Int32 dest bits.
    else if constexpr (BINOP == BinaryOp::EQ_INT || BINOP == BinaryOp::NE_INT)
    {
        constexpr SfpuType comp_type = (BINOP == BinaryOp::EQ_INT) ? SfpuType::eq : SfpuType::ne;
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_eq_int,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS, comp_type, DataFormat::Int32),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    // Integer max/min via SFPSWAP. IS_UNSIGNED selects int32 vs uint32 handling; the
    // sign-magnitude dest only round-trips non-negative results, so the tests keep
    // operands non-negative.
    else if constexpr (BINOP == BinaryOp::MAX_INT32 || BINOP == BinaryOp::MIN_INT32 || BINOP == BinaryOp::MAX_UINT32 || BINOP == BinaryOp::MIN_UINT32)
    {
        constexpr bool IS_MAX      = (BINOP == BinaryOp::MAX_INT32 || BINOP == BinaryOp::MAX_UINT32);
        constexpr bool IS_UNSIGNED = (BINOP == BinaryOp::MAX_UINT32 || BINOP == BinaryOp::MIN_UINT32);
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binary_max_min_int32,
            (IS_MAX, IS_UNSIGNED, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_INT32)
    {
        // int32 remainder r = a - b * trunc(a/b) with sign following the dividend a.
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_remainder_int32,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_UINT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_remainder_uint32,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::FMOD_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_fmod_int32,
            (APPROXIMATION_MODE, PER_FACE_ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

// To add a new metal ternary SFPU operation:
// 1. Include the metal header above: #include "llk_sfpu/<operation>.h"
// 2. Add the operation enum to SfpuType in llk_sfpu_types.h
// 3. Add the if constexpr branches in call_ternary_sfpu_operation_init() and
//    call_ternary_sfpu_operation() below.

/**
 * Calls only the init portion of a ternary SFPU operation.
 * Must be paired with a subsequent call_ternary_sfpu_operation() for the calculate step.
 * Delegates to the SFPU_TERNARY_INIT* macros (llk_math_eltwise_ternary_sfpu_macros.h),
 * which run _llk_math_eltwise_ternary_sfpu_init_<OP>() and the optional per-op init.
 */
template <SfpuType OPERATION, bool APPROX_MODE, bool is_fp32_dest_acc_en>
void call_ternary_sfpu_operation_init()
{
    if constexpr (OPERATION == SfpuType::where)
    {
        SFPU_TERNARY_INIT_FN(where, sfpu::_init_where_, (APPROX_MODE));
    }
    else if constexpr (OPERATION == SfpuType::addcmul)
    {
        // addcmul has no per-op init beyond the shared addrmod setup.
        SFPU_TERNARY_INIT(addcmul);
    }
    else if constexpr (OPERATION == SfpuType::addcdiv)
    {
        // addcdiv uses sfpu_reciprocal internally; init_addcdiv forwards to sfpu_reciprocal_init.
        SFPU_TERNARY_INIT_FN(addcdiv, sfpu::init_addcdiv, (APPROX_MODE));
    }
    else if constexpr (OPERATION == SfpuType::lerp)
    {
        // lerp has no per-op init beyond the shared addrmod setup.
        SFPU_TERNARY_INIT(lerp);
    }
    else if constexpr (OPERATION == SfpuType::snake_beta)
    {
        // snake_beta uses sfpu_reciprocal internally; snake_beta_init forwards to sfpu_reciprocal_init.
        SFPU_TERNARY_INIT_FN(snake_beta, sfpu::snake_beta_init, (APPROX_MODE));
    }
    else
    {
        // Fail loudly on an unrecognized op instead of silently initializing it
        // as `where` (which would mis-compute). Mirrors call_ternary_sfpu_operation().
        LLK_ASSERT(false, "Unsupported ternary operation init");
    }
}

/**
 * Calls only the calculate portion of a ternary SFPU operation.
 * Must be preceded by a call to call_ternary_sfpu_operation_init() for the same operation.
 */
template <
    DstSync DST_SYNC_MODE,
    bool DST_ACCUM_MODE,
    SfpuType OPERATION,
    bool APPROX_MODE,
    bool is_fp32_dest_acc_en,
    DataFormat MATH_FORMAT,
    int ITERATIONS = 8>
void call_ternary_sfpu_operation(
    const std::uint32_t dst_index_in0 = 0,
    const std::uint32_t dst_index_in1 = 1,
    const std::uint32_t dst_index_in2 = 2,
    const std::uint32_t dst_index_out = 0,
    const std::uint32_t value         = 0x40000000u /* 2.0f */,
    ckernel::VectorMode vector_mode   = ckernel::VectorMode::RC)
{
    if constexpr (OPERATION == SfpuType::where)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            _calculate_where_,
            (APPROX_MODE, MATH_FORMAT, ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::addcmul)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_addcmul,
            (APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            vector_mode,
            value);
    }
    else if constexpr (OPERATION == SfpuType::addcdiv)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_addcdiv,
            (APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            vector_mode,
            value);
    }
    else if constexpr (OPERATION == SfpuType::lerp)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_lerp,
            (APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::snake_beta)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_snake_beta,
            (APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS),
            dst_index_in0,
            dst_index_in1,
            dst_index_in2,
            dst_index_out,
            vector_mode);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported ternary operation");
    }
}

} // namespace test_utils
