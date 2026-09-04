// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_add_top_row.h"
#include "llk_sfpu/ckernel_sfpu_topk.h"
#include "llk_sfpu/llk_math_eltwise_sfpu_op.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "sfpu_test_ops.h"

// To add a new metal SFPU operation:
// 1. Include the metal header below: #include "llk_sfpu/<operation>.h"
// 2. Add the enumerator to SfpuUnaryOp in sfpu_test_ops.h
// 3. Add the struct / SfpuUnaryFn adapter to call_unary_sfpu_operation_init() and
//    call_unary_sfpu_operation() below
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
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_where.h"
// Metal comparison-to-zero / unary-int-compare kernels (calculate_comp,
// calculate_comp_int, calculate_comp_uint16, calculate_eqz_uint32,
// calculate_nez_uint32, calculate_comp_unary_int + their *_init). Distinct from
// the tt-llk sfpu/ckernel_sfpu_comp.h (_calculate_zero_comp_ etc.) included
// below; the two share no symbol names so both can coexist.
#include "llk_sfpu/ckernel_sfpu_binop_with_unary.h"
#include "llk_sfpu/ckernel_sfpu_celu.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_digamma.h"
#include "llk_sfpu/ckernel_sfpu_div_int32.h"
#include "llk_sfpu/ckernel_sfpu_div_int32_floor.h"
#include "llk_sfpu/ckernel_sfpu_elu.h"
#include "llk_sfpu/ckernel_sfpu_erf.h"
#include "llk_sfpu/ckernel_sfpu_erfc.h"
#include "llk_sfpu/ckernel_sfpu_erfinv.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_exp2.h"
#include "llk_sfpu/ckernel_sfpu_expm1.h"
#include "llk_sfpu/ckernel_sfpu_fmod.h"
#include "llk_sfpu/ckernel_sfpu_gcd.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_hardmish.h"
#include "llk_sfpu/ckernel_sfpu_hardshrink.h"
#include "llk_sfpu/ckernel_sfpu_hardtanh.h"
#include "llk_sfpu/ckernel_sfpu_heaviside.h"
#include "llk_sfpu/ckernel_sfpu_i0.h"
#include "llk_sfpu/ckernel_sfpu_i1.h"
#include "llk_sfpu/ckernel_sfpu_identity.h"
#include "llk_sfpu/ckernel_sfpu_isclose.h"
#include "llk_sfpu/ckernel_sfpu_lcm.h"
#include "llk_sfpu/ckernel_sfpu_lerp.h"
#include "llk_sfpu/ckernel_sfpu_lgamma.h"
#include "llk_sfpu/ckernel_sfpu_log.h"
#include "llk_sfpu/ckernel_sfpu_log1p.h"
#include "llk_sfpu/ckernel_sfpu_logical_not.h"
#include "llk_sfpu/ckernel_sfpu_logsigmoid.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_mish.h"
#include "llk_sfpu/ckernel_sfpu_mul_int32.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "llk_sfpu/ckernel_sfpu_polygamma.h"
#include "llk_sfpu/ckernel_sfpu_prelu.h"
#include "llk_sfpu/ckernel_sfpu_rdiv.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"
#include "llk_sfpu/ckernel_sfpu_remainder.h"
#include "llk_sfpu/ckernel_sfpu_rpow.h"
#include "llk_sfpu/ckernel_sfpu_rsqrt.h"
#include "llk_sfpu/ckernel_sfpu_rsub_int32.h"
#include "llk_sfpu/ckernel_sfpu_selu.h"
#include "llk_sfpu/ckernel_sfpu_shift.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid_appx.h"
#include "llk_sfpu/ckernel_sfpu_sign.h"
#include "llk_sfpu/ckernel_sfpu_signbit.h"
#include "llk_sfpu/ckernel_sfpu_silu.h"
#include "llk_sfpu/ckernel_sfpu_snake_beta.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_softshrink.h"
#include "llk_sfpu/ckernel_sfpu_softsign.h"
#include "llk_sfpu/ckernel_sfpu_sqrt.h"
#include "llk_sfpu/ckernel_sfpu_sqrt_custom.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_tanh_derivative.h"
#include "llk_sfpu/ckernel_sfpu_tanhshrink.h"
#include "llk_sfpu/ckernel_sfpu_trigonometry.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "llk_sfpu/ckernel_sfpu_unary_comp.h"
#include "llk_sfpu/ckernel_sfpu_unary_max_min.h"
#include "llk_sfpu/ckernel_sfpu_unary_power.h"
#include "llk_sfpu/ckernel_sfpu_unary_shift.h"
#include "llk_sfpu/ckernel_sfpu_xielu.h"
#include "sfpu/ckernel_sfpu_add_int.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_isinf_isnan.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "sfpu/ckernel_sfpu_sub_int.h"
#include "sfpu/ckernel_sfpu_tanh_derivative.h"
#include "sfpu/ckernel_sfpu_threshold.h"

// Test-only SFPU loop/adapter wrappers (calculate_sqrt_custom, calculate_expm1_cw,
// calculate_mask_binary) used by the dispatch below.
#include "sfpu_test_helpers.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, bool IS_FP32_DEST_ACC_EN, int ITERATIONS, bool CLAMP_NEGATIVE, std::uint32_t EXP_BASE_SCALE_FACTOR>
inline __attribute__((always_inline)) void calculate_exponential_const_scale()
{
    calculate_exponential<APPROXIMATION_MODE, IS_FP32_DEST_ACC_EN, true /* SCALE_EN */, ITERATIONS, CLAMP_NEGATIVE>(EXP_BASE_SCALE_FACTOR);
}

} // namespace ckernel::sfpu

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::sfpu;

template <auto>
inline constexpr bool unhandled_sfpu_test_op = false;

template <SfpuUnaryOp OPERATION>
inline constexpr bool is_zero_comp_unary_op()
{
    return OPERATION == SfpuUnaryOp::equal_zero || OPERATION == SfpuUnaryOp::not_equal_zero || OPERATION == SfpuUnaryOp::less_than_zero ||
           OPERATION == SfpuUnaryOp::greater_than_zero || OPERATION == SfpuUnaryOp::less_than_equal_zero || OPERATION == SfpuUnaryOp::greater_than_equal_zero;
}

template <SfpuUnaryOp OPERATION>
constexpr ZeroCompMode zero_comp_mode_of()
{
    if constexpr (OPERATION == SfpuUnaryOp::equal_zero)
    {
        return ZeroCompMode::EqZ;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::not_equal_zero)
    {
        return ZeroCompMode::NeZ;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::less_than_zero)
    {
        return ZeroCompMode::LtZ;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::greater_than_equal_zero)
    {
        return ZeroCompMode::GeZ;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::greater_than_zero)
    {
        return ZeroCompMode::GtZ;
    }
    else
    {
        return ZeroCompMode::LeZ;
    }
}

template <SfpuUnaryOp OPERATION>
constexpr IsInfNanMode isinf_nan_mode_of()
{
    if constexpr (OPERATION == SfpuUnaryOp::isinf)
    {
        return IsInfNanMode::IsInf;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::isposinf)
    {
        return IsInfNanMode::IsPosInf;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::isneginf)
    {
        return IsInfNanMode::IsNegInf;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::isnan)
    {
        return IsInfNanMode::IsNan;
    }
    else
    {
        return IsInfNanMode::IsFinite;
    }
}

template <SfpuUnaryOp OPERATION>
constexpr TrigOp trig_op_of()
{
    if constexpr (OPERATION == SfpuUnaryOp::sine)
    {
        return TrigOp::Sine;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cosine)
    {
        return TrigOp::Cosine;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tan)
    {
        return TrigOp::Tan;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::asin)
    {
        return TrigOp::Asin;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::acos)
    {
        return TrigOp::Acos;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::atan)
    {
        return TrigOp::Atan;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sinh)
    {
        return TrigOp::Sinh;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cosh)
    {
        return TrigOp::Cosh;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::asinh)
    {
        return TrigOp::Asinh;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::acosh)
    {
        return TrigOp::Acosh;
    }
    else
    {
        return TrigOp::Atanh;
    }
}

//
// SFPU typecast dispatch.
//
// Unlike the other unary SFPU operations (which are keyed by a single SfpuUnaryOp),
// typecast selects one of ~25 `calculate_typecast_*` primitives based on the
// (input, output) DataFormat pair. The two helpers below are a faithful,
// parametrized mirror of the production compute API `typecast_tile<IN, OUT>` /
// `typecast_tile_init<IN, OUT>` (tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h),
// rewritten to take explicit template parameters instead of the ambient
// compute-kernel macros. Init goes through sfpu::Typecast<IN,OUT,...>::init();
// calculate still uses the per-pair SfpuUnaryFn adapters. Pairs realised purely
// by unpacker/packer format conversion issue no SFPU calculate call.
//
// These are reached through the shared `call_unary_sfpu_operation[_init]`
// dispatch below via `SfpuUnaryOp::typecast` (with IN/OUT supplied as the trailing
// template parameters). Keep this dispatch in lockstep with typecast.h.
//
template <DataFormat IN, DataFormat OUT, bool APPROX_MODE, bool DST_ACCUM_MODE>
void call_unary_typecast_operation_init()
{
    // Typecast::init() runs the shared SFPU init and then the pair-specific
    // init_typecast_* (or nothing, when the conversion is unpacker/packer-only).
    sfpu::Typecast<IN, OUT, APPROX_MODE, DstSync::SyncHalf, DST_ACCUM_MODE>::init();
}

template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, DataFormat IN, DataFormat OUT, bool APPROX_MODE, int ITERATIONS = 8>
void call_unary_typecast_operation(std::uint32_t dst_index)
{
    if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint16<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float16_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint16_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float16_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_int32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::Int32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Float16_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint16<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint16_to_fp32<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Int32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_int32_to_fp32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint16<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Bfp8_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint16_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::Int32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Bfp8_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_int32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::UInt32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float16_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::UInt32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_fp32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::UInt32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Bfp8_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::UInt32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint16_to_uint32<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Int32)
    {
        // Calls same kernel as the UInt32 case.
        SfpuUnaryFn<sfpu::calculate_typecast_uint16_to_uint32<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_uint16<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_int32_to_uint16<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint16<APPROX_MODE, ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Bfp4_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint16_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::Int32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Bfp4_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_int32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::UInt32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Bfp4_b)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt8)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_fp32_to_uint8<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr ((IN == DataFormat::Int32 || IN == DataFormat::UInt32 || IN == DataFormat::UInt16) && OUT == DataFormat::UInt8)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint_to_uint8<APPROX_MODE, ITERATIONS, (IN == DataFormat::UInt16)>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::Float32)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_fp32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_fp16b<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::UInt16)
    {
        SfpuUnaryFn<sfpu::calculate_typecast_uint32_to_uint16<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
}

/**
 * Calls only the init portion of a unary SFPU operation.
 * Must be paired with a subsequent call_unary_sfpu_operation() for the calculate step.
 *
 * Prefers the production op struct's ::init() when that struct's init_kernel matches
 * the old wrapper callback; otherwise SfpuUnaryFn<kernel, DST_SYNC, DST_ACCUM, init_fn>::init()
 * or the shared _llk_math_eltwise_sfpu_init_() for ops that only needed the generic setup.
 *
 * @tparam OPERATION Test-only SfpuUnaryOp key (emitted by the Python harness)
 * @tparam APPROX_MODE Whether to use approximation mode for the SFPU operation
 * @tparam is_fp32_dest_acc_en Whether the destination accumulator is in FP32 mode
 * @tparam ITERATIONS Number of SFPU iterations (typically 32 for full tile)
 */
template <
    SfpuUnaryOp OPERATION,
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
    // Once-per-kernel SFPU init (SFPU config reg + invariant ADDR_MOD_7). In metal this is hoisted into the
    // full-init entry points (compute_kernel_hw_startup / init_sfpu / unary_op_init_common); this standalone
    // LLK harness bypasses those, so it must run the once-init itself. Op-struct ::init() also runs the
    // shared _llk_math_eltwise_sfpu_init_(); re-running the invariant here is harmless (idempotent).
    _llk_math_eltwise_sfpu_init_();

    constexpr DstSync INIT_SYNC = DstSync::SyncHalf;

    if constexpr (
        OPERATION == SfpuUnaryOp::acosh || OPERATION == SfpuUnaryOp::asinh || OPERATION == SfpuUnaryOp::atanh || OPERATION == SfpuUnaryOp::cosine ||
        OPERATION == SfpuUnaryOp::tan || OPERATION == SfpuUnaryOp::atan || OPERATION == SfpuUnaryOp::asin || OPERATION == SfpuUnaryOp::acos ||
        OPERATION == SfpuUnaryOp::sinh || OPERATION == SfpuUnaryOp::cosh || OPERATION == SfpuUnaryOp::sine)
    {
        sfpu::Trigonometry<trig_op_of<OPERATION>(), APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::exp2)
    {
        sfpu::Exp2<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::exponential || OPERATION == SfpuUnaryOp::exp_with_base)
    {
        sfpu::Exp<APPROX_MODE, CLAMP_NEGATIVE, INIT_SYNC, is_fp32_dest_acc_en, false /* SCALE_EN */, ITERATIONS, 0x3F800000u>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::erfinv)
    {
        sfpu::Erfinv<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::erf)
    {
        sfpu::Erf<APPROX_MODE, false /* IS_ERFC */, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::erfc)
    {
        sfpu::Erf<APPROX_MODE, true /* IS_ERFC */, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::expm1)
    {
        sfpu::Expm1<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cbrt)
    {
        sfpu::Cbrt<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::i1)
    {
        sfpu::I1<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::signbit)
    {
        sfpu::Signbit<APPROX_MODE, DataFormat::Float16_b, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::lgamma)
    {
        sfpu::LgammaStirling<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::digamma)
    {
        sfpu::Digamma<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::fmod)
    {
        sfpu::Fmod<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init(0x40000000u /* 2.0f */, 0x3f000000u /* 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::remainder)
    {
        sfpu::Remainder<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init(0x40000000u /* 2.0f */, 0x3f000000u /* 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rpow)
    {
        sfpu::Rpow<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::power)
    {
        sfpu::UnaryPower<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_max)
    {
        sfpu::UnaryMaxMin<true, DataFormat::Float16_b, APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_min)
    {
        sfpu::UnaryMaxMin<false, DataFormat::Float16_b, APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_max_int32)
    {
        sfpu::UnaryMaxMin<true, DataFormat::Int32, APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_min_int32)
    {
        sfpu::UnaryMaxMin<false, DataFormat::Int32, APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_max_uint32)
    {
        sfpu::UnaryMaxMin<true, DataFormat::UInt32, APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_min_uint32)
    {
        sfpu::UnaryMaxMin<false, DataFormat::UInt32, APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::polygamma)
    {
        sfpu::Polygamma<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::xielu)
    {
        sfpu::Xielu<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sigmoid_appx)
    {
        SfpuUnaryFn<sfpu::calculate_sigmoid_appx<ITERATIONS>, INIT_SYNC, is_fp32_dest_acc_en, sfpu::sigmoid_appx_init>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sigmoid)
    {
        sfpu::Sigmoid<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::mish)
    {
        sfpu::Mish<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rdiv)
    {
        sfpu::Rdiv<APPROX_MODE, ckernel::RoundingMode::None, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu)
    {
        sfpu::Gelu<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu_appx)
    {
        sfpu::Gelu<true, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu_derivative)
    {
        sfpu::GeluDerivative<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::softsign)
    {
        sfpu::Softsign<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu_tanh)
    {
        sfpu::GeluTanh<INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::hardsigmoid)
    {
        sfpu::Activation<APPROX_MODE, ckernel::ActivationType::Hardsigmoid, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::log || OPERATION == SfpuUnaryOp::log_with_base)
    {
        sfpu::Log<APPROX_MODE, FAST_MODE, OPERATION == SfpuUnaryOp::log_with_base, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::silu)
    {
        sfpu::Silu<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::log1p)
    {
        sfpu::Log1p<APPROX_MODE, FAST_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::reciprocal)
    {
        sfpu::Recip<APPROX_MODE, false /* legacy_compat */, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::reciprocal_compat)
    {
        sfpu::Recip<APPROX_MODE, true /* legacy_compat */, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rsqrt)
    {
        sfpu::Rsqrt<APPROX_MODE, FAST_MODE, false /* legacy_compat */, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sqrt)
    {
        sfpu::Sqrt<APPROX_MODE, FAST_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh)
    {
        sfpu::Tanh<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanhshrink)
    {
        sfpu::Tanhshrink<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh_derivative)
    {
        sfpu::TanhDerivative<APPROX_MODE, INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh_derivative_lut)
    {
        SfpuUnaryFn<sfpu::_calculate_tanh_derivative_<APPROX_MODE, 0, ITERATIONS>, INIT_SYNC, is_fp32_dest_acc_en, sfpu::tanh_derivative_init<APPROX_MODE>>::
            init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::typecast)
    {
        call_unary_typecast_operation_init<TYPECAST_IN, TYPECAST_OUT, APPROX_MODE, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::topk_local_sort)
    {
        sfpu::TopkLocalSort<APPROX_MODE, STABLE_SORT, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::topk_merge)
    {
        sfpu::TopkMerge<APPROX_MODE, false, STABLE_SORT, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::topk_rebuild)
    {
        sfpu::TopkRebuild<APPROX_MODE, STABLE_SORT, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (is_zero_comp_unary_op<OPERATION>())
    {
        sfpu::ZeroComp<APPROX_MODE, DataFormat::Float16_b, zero_comp_mode_of<OPERATION>(), INIT_SYNC, is_fp32_dest_acc_en, ITERATIONS>::init();
    }
    else
    {
        // Reset-only / no extra hardware-state ops (floor/ceil/add1/identity/softplus/...).
        // The shared init already resets dest RWC counters.
        _llk_math_eltwise_sfpu_init_();
    }
}

/**
 * Calls only the calculate portion of a unary SFPU operation.
 * Must be preceded by a call to call_unary_sfpu_operation_init() for the same operation.
 * Delegates to SfpuUnaryFn from llk_math_eltwise_sfpu_op.h,
 * which runs the ckernel::_sfpu_check_<DST_SYNC_MODE, DST_ACCUM_MODE>
 * dst-bound LLK_ASSERT and then dispatches directly to
 * _llk_math_eltwise_unary_sfpu_params_. Face-looping versus single-call
 * behavior is selected by the explicit vector_mode parameter;
 * the default/non-face mode preserves the existing single-call full-tile behavior.
 *
 * DST_SYNC_MODE and DST_ACCUM_MODE are the first two template parameters, mirroring
 * the convention of the underlying SFPU macros and helpers (SfpuUnaryFn,
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
    SfpuUnaryOp OPERATION,
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
    // Fixed dispatch constants shared with the golden (golden_generators.py:
    // _int_maxmin_scalar / _int_shift_amount). The two sides must move together, so
    // keep them named on both to avoid a silent golden desync.
    constexpr std::uint32_t MAXMIN_SCALAR = 1000u;
    // Shift amount for the unary shift ops. Overridable from params.h via the SFPU_SHIFT_AMOUNT
    // template parameter so the Python side can sweep it; a test that does not set it keeps the
    // original fixed 3. The golden reads the same value through UnarySFPUGolden's shift_amount
    // argument, so the two sides move together.
#ifdef SFPU_SHIFT_AMOUNT
    constexpr std::uint32_t SHIFT_AMOUNT = SFPU_SHIFT_AMOUNT;
#else
    constexpr std::uint32_t SHIFT_AMOUNT = 3u;
#endif
    // Integer scalar that unary_eq/unary_ne (Int32) compare against via metal
    // calculate_comp_unary_int. Shared with the golden (golden_generators.py:
    // _unary_comp_int_scalar); the two sides must move together.
    constexpr int UNARY_COMP_INT_SCALAR = 5;
    // Clamp/Hardtanh fp32-encoded bounds. Shared with the golden
    // (sfpu_dispatch_constants.py: CLAMP_MIN / CLAMP_MAX); the two sides must move together.
    constexpr std::uint32_t CLAMP_MIN_FP32 = 0xBF800000u; // -1.0f
    constexpr std::uint32_t CLAMP_MAX_FP32 = 0x3F800000u; //  1.0f

    if constexpr (OPERATION == SfpuUnaryOp::abs)
    {
        SfpuUnaryFn<sfpu::calculate_abs<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::abs_int32)
    {
        SfpuUnaryFn<sfpu::calculate_abs_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::add1)
    {
        SfpuUnaryFn<sfpu::calculate_add1<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cast_fp32_to_fp16a)
    {
        SfpuUnaryFn<sfpu::cast_fp32_to_fp16a<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::acosh)
    {
        SfpuUnaryFn<sfpu::calculate_acosh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::asinh)
    {
        SfpuUnaryFn<sfpu::calculate_asinh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::atanh)
    {
        SfpuUnaryFn<sfpu::calculate_atanh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::celu)
    {
        SfpuUnaryFn<sfpu::calculate_celu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f800000u /* alpha = 1.0f */, 0x3f800000u /* 1/alpha = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cosine)
    {
        SfpuUnaryFn<sfpu::calculate_cosine<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tan)
    {
        SfpuUnaryFn<sfpu::calculate_tangent<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::atan)
    {
        SfpuUnaryFn<sfpu::calculate_atan<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::asin)
    {
        SfpuUnaryFn<sfpu::calculate_asin<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::acos)
    {
        SfpuUnaryFn<sfpu::calculate_acos<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sinh)
    {
        SfpuUnaryFn<sfpu::calculate_sinh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cosh)
    {
        SfpuUnaryFn<sfpu::calculate_cosh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::elu)
    {
        SfpuUnaryFn<sfpu::calculate_elu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f800000u /* alpha = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::exp2)
    {
        SfpuUnaryFn<sfpu::calculate_exp2<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    // VectorMode::RC: params drives 4 face iterations with 2×SETRWC between each —
    // the lambda processes 8 rows per face, giving 32 total.
    else if constexpr (OPERATION == SfpuUnaryOp::exponential && APPROX_MODE && CLAMP_NEGATIVE)
    {
        SfpuUnaryFn<sfpu::calculate_exponential<APPROX_MODE, is_fp32_dest_acc_en, false /* scale_en */, 8, CLAMP_NEGATIVE>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index, VectorMode::RC, p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    // Single call (else branch): calculate_exponential handles 8 or 32 iterations internally.
    else if constexpr (OPERATION == SfpuUnaryOp::exponential)
    {
        SfpuUnaryFn<
            sfpu::calculate_exponential<APPROX_MODE, is_fp32_dest_acc_en, false /* scale_en */, ITERATIONS, CLAMP_NEGATIVE>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index, vector_mode, p_sfpu::kCONST_1_FP16B /* exp_base_scale_factor */);
    }
    // exp_with_base = b^x = exp(x * ln b): the only op that drives calculate_exponential
    // with SCALE_EN=true. The bf16 scale 0x3F00 == 0.5 selects base b = e^0.5, so the
    // golden is exp(0.5*x); 0.5 is exact in bf16 so no scale-rounding error is added.
    //
    // The bf16-accurate path (_sfpu_exp_21f_bf16_tti_) lowers the scale via TTI_SFPMULI,
    // whose immediate operand must be a compile-time constant. Pass the scale through the
    // test-only adapter's template arguments so SfpuUnaryFn preserves that constness.
    else if constexpr (OPERATION == SfpuUnaryOp::exp_with_base)
    {
        SfpuUnaryFn<
            sfpu::calculate_exponential_const_scale<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, CLAMP_NEGATIVE, 0x3F00u /* bf16(0.5) exp base scale */>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::fill)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::_calculate_fill_int_<APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, static_cast<std::uint32_t>(fill_const_value));
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SfpuUnaryFn<sfpu::_calculate_fill_int_<APPROX_MODE, ckernel::InstrModLoadStore::LO16, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, static_cast<std::uint32_t>(fill_const_value));
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SfpuUnaryFn<sfpu::_calculate_fill_int_<APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, static_cast<std::uint32_t>(fill_const_value));
        }
        else
        {
            SfpuUnaryFn<sfpu::_calculate_fill_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode, fill_const_value);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu)
    {
        SfpuUnaryFn<sfpu::calculate_gelu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu_appx)
    {
        // Directly exercise the LUT approximation kernel (the APPROXIMATION_MODE=true
        // branch of calculate_gelu). Requires the LReg table loaded by gelu_init<true>.
        SfpuUnaryFn<sfpu::calculate_gelu_appx<ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu_derivative)
    {
        SfpuUnaryFn<sfpu::calculate_gelu_derivative_polynomial<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu_tanh)
    {
        SfpuUnaryFn<sfpu::calculate_gelu_tanh<is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::hardsigmoid)
    {
        SfpuUnaryFn<sfpu::calculate_activation<APPROX_MODE, ckernel::ActivationType::Hardsigmoid, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::log)
    {
        SfpuUnaryFn<sfpu::calculate_log<APPROX_MODE, FAST_MODE, false /* HAS_BASE_SCALING */, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index, vector_mode, 0u /* log_base_scale_factor */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::log_with_base)
    {
        SfpuUnaryFn<
            sfpu::calculate_log<APPROX_MODE, FAST_MODE, true /* HAS_BASE_SCALING */, is_fp32_dest_acc_en, ITERATIONS, true /* IS_BASE_TWO */>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index, vector_mode, 0x3FB8AA3Bu /* 1/ln(2) in fp32 -> log2(x) */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::log1p)
    {
        SfpuUnaryFn<sfpu::calculate_log1p<APPROX_MODE, FAST_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::negative)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::_calculate_negative_int_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
        }
        else
        {
            SfpuUnaryFn<sfpu::_calculate_negative_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::reciprocal)
    {
        SfpuUnaryFn<sfpu::calculate_reciprocal<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::reciprocal_compat)
    {
        // Legacy-compat reciprocal (legacy_compat = true routes calculate_reciprocal to
        // _calculate_reciprocal_compat_). Distinct from SfpuUnaryOp::reciprocal, which exercises
        // the accurate legacy_compat = false path. Both are covered because the Compute API's
        // recip_tile()/recip_tile_init() default to legacy_compat = true, so the *default*
        // production path is this one -- and without this op the suite would only ever build
        // the non-default kernel.
        SfpuUnaryFn<sfpu::calculate_reciprocal<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, true /* legacy_compat */>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rsqrt)
    {
        SfpuUnaryFn<sfpu::calculate_rsqrt<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE, false /* legacy_compat */>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::silu)
    {
        SfpuUnaryFn<sfpu::calculate_silu<is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanhshrink)
    {
        SfpuUnaryFn<sfpu::calculate_tanhshrink<is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::floor)
    {
        SfpuUnaryFn<sfpu::_calculate_floor_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::round)
    {
        SfpuUnaryFn<sfpu::_calculate_round_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode, 0 /* decimals */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::ceil)
    {
        SfpuUnaryFn<sfpu::_calculate_ceil_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::trunc)
    {
        SfpuUnaryFn<sfpu::_calculate_trunc_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::frac)
    {
        SfpuUnaryFn<sfpu::_calculate_frac_<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sine)
    {
        SfpuUnaryFn<sfpu::calculate_sine<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sqrt)
    {
        SfpuUnaryFn<sfpu::calculate_sqrt<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::square)
    {
        SfpuUnaryFn<sfpu::calculate_square<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::signbit)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::calculate_signbit_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
        }
        else
        {
            SfpuUnaryFn<sfpu::calculate_signbit<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh)
    {
        SfpuUnaryFn<sfpu::calculate_tanh<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::threshold)
    {
        SfpuUnaryFn<sfpu::_calculate_threshold_<APPROX_MODE, ITERATIONS, float>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 5.0f /* threshold_value */, 10.0f /* replacement_value */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::topk_local_sort)
    {
        SfpuUnaryFn<sfpu::_bitonic_topk_phases_steps<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0 /* idir */, 5 /* i_end_phase */, 0 /* i_start_phase */, 10 /* i_end_step */, 0 /* i_start_step */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::topk_merge)
    {
        SfpuUnaryFn<sfpu::_bitonic_topk_merge<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 5 /* m_iter */, 10 /* k */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::topk_rebuild)
    {
        SfpuUnaryFn<sfpu::_bitonic_topk_rebuild<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, false /* idir */, 5 /* m_iter */, 10 /* k */, 3 /* logk */, 0 /* skip_second */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::relu_max)
    {
        SfpuUnaryFn<sfpu::_relu_max_<sfpi::vFloat, APPROX_MODE, ITERATIONS, float>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 5.0f /* threshold */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::relu_min)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::_relu_min_<sfpi::vInt, APPROX_MODE, ITERATIONS, std::uint32_t>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, 5u /* threshold */);
        }
        else
        {
            SfpuUnaryFn<sfpu::_relu_min_<sfpi::vFloat, APPROX_MODE, ITERATIONS, float>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, 5.0f /* threshold */);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::lrelu)
    {
        SfpuUnaryFn<sfpu::_calculate_lrelu_<APPROX_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, ITERATIONS, 0x3dcccccdu /* slope = 0.1f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::add_int32)
    {
        SfpuUnaryFn<sfpu::calculate_add_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode, 5u /* scalar */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sub_int32)
    {
        SfpuUnaryFn<sfpu::calculate_sub_int32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode, 5u /* scalar */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::bitwise_not)
    {
        SfpuUnaryFn<sfpu::calculate_bitwise_not<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::logical_not_unary)
    {
        // logical_not(x) = (x == 0) ? 1 : 0. Select the layout from the runtime input format,
        if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            SfpuUnaryFn<sfpu::calculate_logical_not<APPROX_MODE, ckernel::InstrModLoadStore::LO16, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::Int32) || math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            SfpuUnaryFn<sfpu::calculate_logical_not<APPROX_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode);
        }
        else
        {
            SfpuUnaryFn<sfpu::calculate_logical_not<APPROX_MODE, ckernel::InstrModLoadStore::DEFAULT, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::heaviside)
    {
        SfpuUnaryFn<sfpu::calculate_heaviside<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::softshrink)
    {
        SfpuUnaryFn<sfpu::calculate_softshrink<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* lambda = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::softsign)
    {
        SfpuUnaryFn<sfpu::calculate_softsign<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sigmoid)
    {
        SfpuUnaryFn<sfpu::calculate_sigmoid<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::mish)
    {
        SfpuUnaryFn<sfpu::calculate_mish<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::selu)
    {
        // selu constants passed as fp32 bit patterns: scale ~= 1.0507, alpha ~= 1.6733.
        SfpuUnaryFn<sfpu::calculate_selu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f867d5fu /* scale */, 0x3fd62d7du /* alpha */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::i0)
    {
        SfpuUnaryFn<sfpu::calculate_i0<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rdiv)
    {
        SfpuUnaryFn<sfpu::calculate_rdiv<APPROX_MODE, is_fp32_dest_acc_en, ckernel::RoundingMode::None, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x40000000u /* value = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::clamp)
    {
        SfpuUnaryFn<sfpu::calculate_clamp<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, CLAMP_MIN_FP32, CLAMP_MAX_FP32);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::hardtanh)
    {
        SfpuUnaryFn<sfpu::calculate_hardtanh<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, CLAMP_MIN_FP32, CLAMP_MAX_FP32);
    }
    else if constexpr (
        OPERATION == SfpuUnaryOp::equal_zero || OPERATION == SfpuUnaryOp::not_equal_zero || OPERATION == SfpuUnaryOp::less_than_zero ||
        OPERATION == SfpuUnaryOp::greater_than_zero || OPERATION == SfpuUnaryOp::less_than_equal_zero || OPERATION == SfpuUnaryOp::greater_than_equal_zero)
    {
        // Comparison-to-zero. The production comp.h API selects a format-specific
        // metal kernel by the input data type (eqz_tile / eqz_tile_int32 /
        // eqz_tile_uint16 / eqz_tile_uint32 ...); mirror that dispatch here so the
        // tt-llk harness exercises the same metal functions in
        // llk_sfpu/ckernel_sfpu_comp.h. The input format arrives as the runtime
        // math_format (see the fill/add_int branches for the same pattern).
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::calculate_comp_int<APPROX_MODE, zero_comp_mode_of<OPERATION>(), ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode);
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt16))
        {
            // uint16 comparison is only defined for eqz/nez (metal static_asserts this).
            if constexpr (OPERATION == SfpuUnaryOp::equal_zero || OPERATION == SfpuUnaryOp::not_equal_zero)
            {
                SfpuUnaryFn<sfpu::calculate_comp_uint16<APPROX_MODE, zero_comp_mode_of<OPERATION>(), ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                    dst_index, vector_mode);
            }
        }
        else if (math_format == ckernel::to_underlying(DataFormat::UInt32))
        {
            // uint32 comparison is only defined for eqz/nez (dedicated metal kernels).
            if constexpr (OPERATION == SfpuUnaryOp::equal_zero)
            {
                SfpuUnaryFn<sfpu::calculate_eqz_uint32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
            }
            else if constexpr (OPERATION == SfpuUnaryOp::not_equal_zero)
            {
                SfpuUnaryFn<sfpu::calculate_nez_uint32<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
            }
        }
        else
        {
            // Float (bf16 / fp32 / bfp8) comparison-to-zero: NaN-aware metal calculate_comp.
            SfpuUnaryFn<sfpu::calculate_comp<APPROX_MODE, zero_comp_mode_of<OPERATION>(), ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode);
        }
    }
    else if constexpr (
        OPERATION == SfpuUnaryOp::isinf || OPERATION == SfpuUnaryOp::isposinf || OPERATION == SfpuUnaryOp::isneginf || OPERATION == SfpuUnaryOp::isnan ||
        OPERATION == SfpuUnaryOp::isfinite)
    {
        // Predicate ops: write 1.0f where the (isinf/isposinf/isneginf/isnan/isfinite)
        // test holds, else 0.0f. The concrete predicate is selected by OPERATION.
        SfpuUnaryFn<sfpu::_calculate_sfpu_isinf_isnan_<isinf_nan_mode_of<OPERATION>(), APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::erfinv)
    {
        SfpuUnaryFn<sfpu::calculate_erfinv<APPROX_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::erf)
    {
        SfpuUnaryFn<sfpu::calculate_erf<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::erfc)
    {
        SfpuUnaryFn<sfpu::calculate_erfc<ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::expm1)
    {
        SfpuUnaryFn<sfpu::calculate_expm1<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cbrt)
    {
        SfpuUnaryFn<sfpu::calculate_cube_root<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::i1)
    {
        SfpuUnaryFn<sfpu::calculate_i1<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sign)
    {
        SfpuUnaryFn<sfpu::calculate_sign<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode, 0u /* exponent_size_8 */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh_derivative)
    {
        SfpuUnaryFn<sfpu::calculate_tanh_derivative_sech2<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh_derivative_lut)
    {
        // Legacy tt-llk primitive: computes 1 - tanh(x)^2 with tanh from the LUT
        // (WITH_PRECOMPUTED_TANH = 0). Distinct from the accurate sech2 variant above.
        SfpuUnaryFn<sfpu::_calculate_tanh_derivative_<APPROX_MODE, 0, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, ITERATIONS);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::hardmish)
    {
        SfpuUnaryFn<sfpu::hardmish<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::lgamma)
    {
        SfpuUnaryFn<sfpu::calculate_lgamma_stirling<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::digamma)
    {
        SfpuUnaryFn<sfpu::calculate_digamma<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::identity)
    {
        SfpuUnaryFn<sfpu::calculate_identity<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::prelu)
    {
        SfpuUnaryFn<sfpu::calculate_prelu<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3e800000u /* slope = 0.25f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rpow)
    {
        SfpuUnaryFn<sfpu::calculate_rpow<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x40000000u /* base = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::power)
    {
        SfpuUnaryFn<sfpu::calculate_unary_power<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x40000000u /* exponent = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::fmod)
    {
        // calculate_fmod() takes no runtime args: it reads vConstFloatPrgm0/1 programmed by
        // init_fmod() (see call_unary_sfpu_operation_init above), so no value/recip is passed.
        SfpuUnaryFn<sfpu::calculate_fmod<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::remainder)
    {
        // Unlike calculate_fmod, calculate_remainder() takes no runtime args: it reads vConstFloatPrgm0/1 programmed
        // by init_remainder() (see call_unary_sfpu_operation_init above), so the call must not pass value/recip.
        SfpuUnaryFn<sfpu::calculate_remainder<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_gt)
    {
        SfpuUnaryFn<sfpu::calculate_unary_gt<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_ne)
    {
        // Int32 input compares against the integer scalar via metal calculate_comp_unary_int
        // (== production unary_ne_tile_int32); float input keeps the fp32 0.5 threshold path.
        // UNARY_COMP_INT_SCALAR is shared with the golden (golden_generators.py:
        // _unary_comp_int_scalar) — keep the two in sync.
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::calculate_comp_unary_int<APPROX_MODE, UnaryCompMode::Ne, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, UNARY_COMP_INT_SCALAR);
        }
        else
        {
            SfpuUnaryFn<sfpu::calculate_unary_ne<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_eq)
    {
        if (math_format == ckernel::to_underlying(DataFormat::Int32))
        {
            SfpuUnaryFn<sfpu::calculate_comp_unary_int<APPROX_MODE, UnaryCompMode::Eq, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, UNARY_COMP_INT_SCALAR);
        }
        else
        {
            SfpuUnaryFn<sfpu::calculate_unary_eq<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
        }
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_lt)
    {
        SfpuUnaryFn<sfpu::calculate_unary_lt<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_ge)
    {
        SfpuUnaryFn<sfpu::calculate_unary_ge<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_le)
    {
        SfpuUnaryFn<sfpu::calculate_unary_le<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_max)
    {
        SfpuUnaryFn<sfpu::calculate_unary_max_min<true, APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0u /* value = 0.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_min)
    {
        SfpuUnaryFn<sfpu::calculate_unary_max_min<false, APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0u /* value = 0.0f */);
    }
    // Integer unary max/min against a fixed scalar (1000). IS_UNSIGNED selects the
    // uint32 vs int32 SFPSWAP handling. The golden compares against the same 1000.
    else if constexpr (OPERATION == SfpuUnaryOp::unary_max_int32)
    {
        SfpuUnaryFn<
            sfpu::calculate_unary_max_min_int32<true /* IS_MAX_OP */, false /* IS_UNSIGNED */, APPROX_MODE, ITERATIONS>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index, vector_mode, MAXMIN_SCALAR);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_min_int32)
    {
        SfpuUnaryFn<
            sfpu::calculate_unary_max_min_int32<false /* IS_MAX_OP */, false /* IS_UNSIGNED */, APPROX_MODE, ITERATIONS>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index, vector_mode, MAXMIN_SCALAR);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_max_uint32)
    {
        SfpuUnaryFn<sfpu::calculate_unary_max_min_int32<true /* IS_MAX_OP */, true /* IS_UNSIGNED */, APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index, vector_mode, MAXMIN_SCALAR);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::unary_min_uint32)
    {
        SfpuUnaryFn<
            sfpu::calculate_unary_max_min_int32<false /* IS_MAX_OP */, true /* IS_UNSIGNED */, APPROX_MODE, ITERATIONS>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index, vector_mode, MAXMIN_SCALAR);
    }
    // Unary shift by a fixed immediate (SHIFT_AMOUNT bits). Integer-only kernels run
    // exclusively on the Int32 path here: the only wiring (test_eltwise_unary_sfpu_int)
    // drives shifts as Int32, and the golden (_left_shift/_right_shift) does unbounded
    // Python integer shifts with no 16-bit masking. UInt16/UInt32 shift branches were
    // dropped as dead+untested; re-add them together with a masked golden if needed.
    else if constexpr (OPERATION == SfpuUnaryOp::left_shift)
    {
        SfpuUnaryFn<sfpu::calculate_left_shift<APPROX_MODE, DataFormat::Int32, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, SHIFT_AMOUNT);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::right_shift)
    {
        SfpuUnaryFn<sfpu::calculate_right_shift<APPROX_MODE, DataFormat::Int32, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, SHIFT_AMOUNT);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::polygamma)
    {
        // order n = 1 (trigamma); scale = (-1)^(n+1) * n! = 1.0f.
        SfpuUnaryFn<sfpu::calculate_polygamma<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f800000u /* n = 1.0f */, 0x3f800000u /* scale = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::xielu)
    {
        SfpuUnaryFn<sfpu::calculate_xielu<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f800000u /* alpha_p = 1.0f */, 0x3f800000u /* alpha_n = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::hardshrink)
    {
        SfpuUnaryFn<sfpu::calculate_hardshrink<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f000000u /* lambda = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::softplus)
    {
        SfpuUnaryFn<sfpu::calculate_softplus<APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index, vector_mode, 0x3f800000u /* beta = 1.0f */, 0x3f800000u /* 1/beta = 1.0f */, 0x41a00000u /* threshold = 20.0f */);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sigmoid_appx)
    {
        SfpuUnaryFn<sfpu::calculate_sigmoid_appx<ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sqrt_custom)
    {
        SfpuUnaryFn<sfpu::calculate_sqrt_custom<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rsqrt_compat)
    {
        // Legacy-compat rsqrt: reciprocal-root method (legacy_compat = true routes
        // calculate_rsqrt to _calculate_rsqrt_compat_). Distinct from SfpuUnaryOp::rsqrt,
        // which exercises the accurate legacy_compat = false path.
        SfpuUnaryFn<sfpu::calculate_rsqrt<APPROX_MODE, ITERATIONS, is_fp32_dest_acc_en, FAST_MODE, true /* legacy_compat */>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::expm1_cw)
    {
        SfpuUnaryFn<sfpu::calculate_expm1_cw<APPROX_MODE, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(dst_index, vector_mode);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::typecast)
    {
        call_unary_typecast_operation<DST_SYNC_MODE, DST_ACCUM_MODE, TYPECAST_IN, TYPECAST_OUT, APPROX_MODE, ITERATIONS>(dst_index);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

template <BinaryOp BINOP>
constexpr BinaryCompMode binary_comp_mode_of()
{
    if constexpr (BINOP == BinaryOp::LT)
    {
        return BinaryCompMode::Lt;
    }
    else if constexpr (BINOP == BinaryOp::GT)
    {
        return BinaryCompMode::Gt;
    }
    else if constexpr (BINOP == BinaryOp::LE)
    {
        return BinaryCompMode::Le;
    }
    else if constexpr (BINOP == BinaryOp::GE)
    {
        return BinaryCompMode::Ge;
    }
    else if constexpr (BINOP == BinaryOp::EQ)
    {
        return BinaryCompMode::Eq;
    }
    else
    {
        return BinaryCompMode::Ne;
    }
}

/**
 * Calls only the init portion of a binary SFPU operation.
 * Must be paired with a subsequent call_binary_sfpu_operation() for the calculate step.
 *
 * DST_ACCUM_MODE (is_fp32_dest_acc_en) must match the value passed to the paired
 * call_binary_sfpu_operation(): ops such as atan2 load an fp32- vs bf16-specific
 * reciprocal polynomial at init time that the calculate step then relies on.
 */
template <bool APPROXIMATION_MODE, bool DST_ACCUM_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation_init()
{
    constexpr DstSync INIT_SYNC = DstSync::SyncHalf;
    if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::DIV || BINOP == BinaryOp::RSUB ||
        BINOP == BinaryOp::XLOGY || BINOP == BinaryOp::POW)
    {
        sfpu::SfpuBinary<APPROXIMATION_MODE, BINOP, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        constexpr DataFormat add_top_row_format = static_cast<DataFormat>(MATH_FORMAT);
        sfpu::AddTopRow<INIT_SYNC, DST_ACCUM_MODE, add_top_row_format>::init();
    }
    else if constexpr (BINOP == BinaryOp::RSHFT || BINOP == BinaryOp::LOGICAL_RSHFT || BINOP == BinaryOp::LSHFT)
    {
        _llk_math_eltwise_sfpu_init_();
    }
    else if constexpr (
        BINOP == BinaryOp::LT || BINOP == BinaryOp::GT || BINOP == BinaryOp::LE || BINOP == BinaryOp::GE || BINOP == BinaryOp::EQ || BINOP == BinaryOp::NE)
    {
        constexpr DataFormat fmt = (MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32)) ? DataFormat::Int32 : DataFormat::Float16_b;
        sfpu::BinaryComp<APPROXIMATION_MODE, binary_comp_mode_of<BINOP>(), fmt, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MAX)
    {
        sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::Float16_b, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MIN)
    {
        sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::Float16_b, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::FMOD)
    {
        sfpu::BinaryFmod<APPROXIMATION_MODE, DataFormat::Float16_b, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER)
    {
        sfpu::BinaryRemainder<APPROXIMATION_MODE, DataFormat::Float16_b, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32)
    {
        sfpu::DivInt32Rounding<APPROXIMATION_MODE, false /* IS_FLOOR */, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32_FLOOR)
    {
        sfpu::DivInt32Rounding<APPROXIMATION_MODE, true /* IS_FLOOR */, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::GCD)
    {
        sfpu::Gcd<INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::LCM)
    {
        sfpu::Lcm<INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::ATAN2)
    {
        sfpu::Atan2<APPROXIMATION_MODE, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MUL_INT32)
    {
        sfpu::MulInt<APPROXIMATION_MODE, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::EQ_INT)
    {
        sfpu::BinaryComp<APPROXIMATION_MODE, BinaryCompMode::Eq, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::NE_INT)
    {
        sfpu::BinaryComp<APPROXIMATION_MODE, BinaryCompMode::Ne, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MAX_INT32)
    {
        sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MIN_INT32)
    {
        sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MAX_UINT32)
    {
        sfpu::BinaryMaxMin<true /* IS_MAX */, DataFormat::UInt32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::MIN_UINT32)
    {
        sfpu::BinaryMaxMin<false /* IS_MAX */, DataFormat::UInt32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_INT32)
    {
        sfpu::BinaryRemainder<APPROXIMATION_MODE, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_UINT32)
    {
        sfpu::BinaryRemainder<APPROXIMATION_MODE, DataFormat::UInt32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::FMOD_INT32)
    {
        sfpu::BinaryFmod<APPROXIMATION_MODE, DataFormat::Int32, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else if constexpr (BINOP == BinaryOp::ISCLOSE)
    {
        sfpu::IsClose<APPROXIMATION_MODE, false /* EQUAL_NAN */, INIT_SYNC, DST_ACCUM_MODE>::init();
    }
    else
    {
        // BITWISE_AND/OR/XOR, RSUB_INT32, MASK, LOGSIGMOID: no extra hardware state.
        _llk_math_eltwise_sfpu_init_();
    }
}

/**
 * Calls only the calculate portion of a binary SFPU operation.
 * Must be preceded by a call to call_binary_sfpu_operation_init() for the same operation.
 * Uses SfpuBinaryFn from llk_math_eltwise_sfpu_op.h, which
 * runs the ckernel::_sfpu_binary_check_<DST_SYNC_MODE, DST_ACCUM_MODE>
 * dst-bound LLK_ASSERTs and then dispatches directly to
 * _llk_math_eltwise_binary_sfpu_params_. The callable receives
 * (dst_index_in0, dst_index_in1, dst_index_out) forwarded from the params
 * wrapper.
 *
 * DST_SYNC_MODE and DST_ACCUM_MODE are the first two template parameters (matching
 * the SfpuBinaryFn / SfpuOpBase::check_dst_index convention) so the dst-bound
 * LLK_ASSERTs run against the kernel's actual sync/accumulation mode.
 */
template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
void call_binary_sfpu_operation(
    const std::uint32_t dst_index_in0 = 0,
    const std::uint32_t dst_index_in1 = 1,
    const std::uint32_t dst_index_out = 0,
    ckernel::VectorMode vector_mode   = ckernel::VectorMode::RC)
{
    // NOTE: The functions invoked via SfpuBinaryFn below run inside
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
        SfpuBinaryFn<sfpu::calculate_sfpu_binary_div<APPROXIMATION_MODE, BINOP, PER_FACE_ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (
        BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL || BINOP == BinaryOp::RSUB || BINOP == BinaryOp::XLOGY ||
        BINOP == BinaryOp::POW)
    {
        if constexpr (BINOP == BinaryOp::ADD && MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32))
        {
            SfpuBinaryFn<
                sfpu::_add_int_<APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32, true /* SIGN_MAGNITUDE_FORMAT */>,
                DST_SYNC_MODE,
                DST_ACCUM_MODE>::calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
        }
        else if constexpr (BINOP == BinaryOp::SUB && MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32))
        {
            // Int32 SUB must use the integer path (_sub_int_); otherwise it would
            // fall through to calculate_sfpu_binary and subtract the raw integer
            // bit-patterns as floats. Mirrors the Int32 ADD path above.
            SfpuBinaryFn<
                sfpu::_sub_int_<APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32, true /* SIGN_MAGNITUDE_FORMAT */>,
                DST_SYNC_MODE,
                DST_ACCUM_MODE>::calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
        }
        else
        {
            SfpuBinaryFn<sfpu::calculate_sfpu_binary<APPROXIMATION_MODE, BINOP, PER_FACE_ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
        }
    }
    else if constexpr (BINOP == BinaryOp::RSHFT)
    {
        SfpuBinaryFn<
            sfpu::calculate_binary_right_shift<APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32_2S_COMP, false>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LSHFT)
    {
        SfpuBinaryFn<
            sfpu::calculate_binary_left_shift<APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32_2S_COMP, false>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LOGICAL_RSHFT)
    {
        SfpuBinaryFn<
            sfpu::calculate_logical_right_shift<APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32_2S_COMP, false>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ADD_TOP_ROW)
    {
        // Use actual format when compiling for ADD_TOP_ROW tests, otherwise use Float32 as safe default for static assert
        constexpr DataFormat add_top_row_format = (BINOP == BinaryOp::ADD_TOP_ROW) ? static_cast<DataFormat>(MATH_FORMAT) : DataFormat::Float32;
        // Force VectorMode::RC_custom so the params wrapper drives all four faces (4 x 8 = 32 rows) of the tile.
        //  _llk_math_eltwise_binary_sfpu_params_ takes its single-call branch
        //  and does not emit the per-face TTI_SETRWC
        SfpuBinaryFn<sfpu::calculate_add_top_row<add_top_row_format>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, ckernel::VectorMode::RC_custom);
    }
    else if constexpr (
        BINOP == BinaryOp::LT || BINOP == BinaryOp::GT || BINOP == BinaryOp::LE || BINOP == BinaryOp::GE || BINOP == BinaryOp::EQ || BINOP == BinaryOp::NE)
    {
        constexpr BinaryCompMode comp_type = binary_comp_mode_of<BINOP>();
        if constexpr (MATH_FORMAT == static_cast<std::uint32_t>(DataFormat::Int32))
        {
            SfpuBinaryFn<sfpu::calculate_binary_comp_int32<APPROXIMATION_MODE, PER_FACE_ITERATIONS, comp_type>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
        }
        else
        {
            SfpuBinaryFn<sfpu::calculate_binary_comp_fp32<APPROXIMATION_MODE, PER_FACE_ITERATIONS, comp_type>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
                dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
        }
    }
    else if constexpr (BINOP == BinaryOp::MAX || BINOP == BinaryOp::MIN)
    {
        // float elementwise max/min (SFPSWAP min/max). Operands read from two dst tiles.
        constexpr bool IS_MAX = (BINOP == BinaryOp::MAX);
        SfpuBinaryFn<sfpu::calculate_binary_max_min<IS_MAX, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::FMOD)
    {
        // float fmod (result sign follows dividend a); DST_ACCUM_MODE selects fp32 vs bf16 store.
        SfpuBinaryFn<sfpu::calculate_sfpu_binary_fmod<APPROXIMATION_MODE, PER_FACE_ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER)
    {
        // float remainder (result sign follows divisor b); DST_ACCUM_MODE selects fp32 vs bf16 store.
        SfpuBinaryFn<sfpu::calculate_sfpu_binary_remainder<APPROXIMATION_MODE, PER_FACE_ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::BITWISE_AND || BINOP == BinaryOp::BITWISE_OR || BINOP == BinaryOp::BITWISE_XOR)
    {
        // int32 bitwise AND/OR/XOR (raw two's-complement bit patterns in dest).
        constexpr BinaryBitwiseOp BW = (BINOP == BinaryOp::BITWISE_AND)  ? BinaryBitwiseOp::AND
                                       : (BINOP == BinaryOp::BITWISE_OR) ? BinaryBitwiseOp::OR
                                                                         : BinaryBitwiseOp::XOR;
        SfpuBinaryFn<
            sfpu::calculate_sfpu_binary_bitwise<APPROXIMATION_MODE, BW, ckernel::InstrModLoadStore::INT32, PER_FACE_ITERATIONS>,
            DST_SYNC_MODE,
            DST_ACCUM_MODE>::calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32)
    {
        // int32 truncating division (rounds toward zero). calculate_div_int32_trunc writes a
        // true int32 quotient; the legacy calculate_div_int32 stored an fp32 result, which the
        // Int32 pack path reinterpreted as garbage bit patterns.
        SfpuBinaryFn<sfpu::calculate_div_int32_trunc<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::DIV_INT32_FLOOR)
    {
        // int32 floor division (rounds toward -inf).
        SfpuBinaryFn<sfpu::calculate_div_int32_floor<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::GCD)
    {
        // int32 gcd via the binary-GCD REPLAY loop recorded in gcd_init.
        SfpuBinaryFn<sfpu::calculate_sfpu_gcd<PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::LCM)
    {
        // int32 lcm = a/gcd(a,b) * b (binary-GCD + reciprocal); operands assumed < 2^15.
        SfpuBinaryFn<sfpu::calculate_sfpu_lcm<PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::RSUB_INT32)
    {
        SfpuBinaryFn<sfpu::calculate_rsub_int<APPROXIMATION_MODE, ckernel::InstrModLoadStore::INT32, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::MASK)
    {
        // float mask: out = (mask != 0) ? data : 0, with data at in0 and mask at in1.
        // Driven through the test-only adapter since calculate_mask uses fixed dst
        // offsets rather than the forwarded indices.
        SfpuBinaryFn<sfpu::calculate_mask_binary<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ATAN2)
    {
        // atan2(y, x): in0 = y, in1 = x (calculate_sfpu_atan2 forwards them as
        // _sfpu_atan2_(in0, in1)). DST_ACCUM_MODE is is_fp32_dest_acc_en and selects
        // the higher-order fp32 minimax polynomial (vs the bf16 one) plus the final
        // convert-to-bf16 rounding, so it must match the init's variant.
        SfpuBinaryFn<sfpu::calculate_sfpu_atan2<APPROXIMATION_MODE, PER_FACE_ITERATIONS, DST_ACCUM_MODE>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::MUL_INT32)
    {
        // int32 multiply: out = in0 * in1 (low 32 bits). The kernel loads/stores via
        // plain INT32 (two's-complement dest bits), so the sign-magnitude packer only
        // round-trips non-negative results; the test keeps operands positive with a
        // product < 2^31 (see test_eltwise_binary_sfpu_int_uniform).
        SfpuBinaryFn<sfpu::mul_int32<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::ISCLOSE)
    {
        // isclose: out = (|a - b| <= atol + rtol * |b|) ? 1 : 0, with a=in0, b=in1.
        // rtol/atol are passed as fp32 bit patterns via the params wrapper's runtime-arg
        // forwarding. Fixed to torch's defaults rtol=1e-5 (0x3727c5ac), atol=1e-8
        // (0x322bcc77); EQUAL_NAN=false, so any NaN operand yields 0. The test uses
        // large-margin stimuli so the exact tolerance (and fp32-vs-bf16 rounding of the
        // tol term) never flips the pass/fail decision.
        SfpuBinaryFn<sfpu::calculate_sfpu_isclose<APPROXIMATION_MODE, PER_FACE_ITERATIONS, /*EQUAL_NAN=*/false>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode, /*rtol_bits=*/0x3727c5acu, /*atol_bits=*/0x322bcc77u);
    }
    else if constexpr (BINOP == BinaryOp::LOGSIGMOID)
    {
        // logsigmoid(x) = -softplus(-x), with x = in0 and exp(-x) = in1 (the compute
        // kernel is expected to supply exp(-x) as the second operand; the test bakes
        // it into the paired stimuli). No dedicated init (baseline add1 addrmod).
        SfpuBinaryFn<sfpu::calculate_logsigmoid<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    // Integer relational eq/ne: XOR-based exact compare over Int32 dest bits.
    else if constexpr (BINOP == BinaryOp::EQ_INT || BINOP == BinaryOp::NE_INT)
    {
        constexpr BinaryCompMode comp_type = (BINOP == BinaryOp::EQ_INT) ? BinaryCompMode::Eq : BinaryCompMode::Ne;
        SfpuBinaryFn<sfpu::calculate_binary_eq_int<APPROXIMATION_MODE, PER_FACE_ITERATIONS, comp_type, DataFormat::Int32>, DST_SYNC_MODE, DST_ACCUM_MODE>::
            calculate(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    // Integer max/min via SFPSWAP. IS_UNSIGNED selects int32 vs uint32 handling; the
    // sign-magnitude dest only round-trips non-negative results, so the tests keep
    // operands non-negative.
    else if constexpr (BINOP == BinaryOp::MAX_INT32 || BINOP == BinaryOp::MIN_INT32 || BINOP == BinaryOp::MAX_UINT32 || BINOP == BinaryOp::MIN_UINT32)
    {
        constexpr bool IS_MAX      = (BINOP == BinaryOp::MAX_INT32 || BINOP == BinaryOp::MAX_UINT32);
        constexpr bool IS_UNSIGNED = (BINOP == BinaryOp::MAX_UINT32 || BINOP == BinaryOp::MIN_UINT32);
        SfpuBinaryFn<sfpu::calculate_binary_max_min_int32<IS_MAX, IS_UNSIGNED, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_INT32)
    {
        // int32 remainder r = a - b * trunc(a/b) with sign following the dividend a.
        SfpuBinaryFn<sfpu::calculate_remainder_int32<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::REMAINDER_UINT32)
    {
        SfpuBinaryFn<sfpu::calculate_remainder_uint32<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (BINOP == BinaryOp::FMOD_INT32)
    {
        SfpuBinaryFn<sfpu::calculate_fmod_int32<APPROXIMATION_MODE, PER_FACE_ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported operation");
    }
}

// To add a new metal ternary SFPU operation:
// 1. Include the metal header above: #include "llk_sfpu/<operation>.h"
// 2. Add the enumerator to SfpuTernaryOp in sfpu_test_ops.h
// 3. Add the if constexpr branches in call_ternary_sfpu_operation_init() and
//    call_ternary_sfpu_operation() below.

/**
 * Calls only the init portion of a ternary SFPU operation.
 * Must be paired with a subsequent call_ternary_sfpu_operation() for the calculate step.
 */
template <SfpuTernaryOp OPERATION, bool APPROX_MODE, bool is_fp32_dest_acc_en>
void call_ternary_sfpu_operation_init()
{
    constexpr DstSync INIT_SYNC = DstSync::SyncHalf;
    if constexpr (OPERATION == SfpuTernaryOp::where)
    {
        sfpu::Where<APPROX_MODE, DataFormat::Float16_b, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuTernaryOp::addcmul)
    {
        sfpu::Addcmul<APPROX_MODE, DataFormat::Float16_b, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuTernaryOp::addcdiv)
    {
        sfpu::Addcdiv<APPROX_MODE, DataFormat::Float16_b, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuTernaryOp::lerp)
    {
        sfpu::Lerp<APPROX_MODE, DataFormat::Float16_b, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else if constexpr (OPERATION == SfpuTernaryOp::snake_beta)
    {
        sfpu::SnakeBeta<APPROX_MODE, DataFormat::Float16_b, INIT_SYNC, is_fp32_dest_acc_en>::init();
    }
    else
    {
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
    SfpuTernaryOp OPERATION,
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
    if constexpr (OPERATION == SfpuTernaryOp::where)
    {
        SfpuTernaryFn<sfpu::_calculate_where_<APPROX_MODE, MATH_FORMAT, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, vector_mode);
    }
    else if constexpr (OPERATION == SfpuTernaryOp::addcmul)
    {
        SfpuTernaryFn<sfpu::calculate_addcmul<APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, vector_mode, value);
    }
    else if constexpr (OPERATION == SfpuTernaryOp::addcdiv)
    {
        SfpuTernaryFn<sfpu::calculate_addcdiv<APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, vector_mode, value);
    }
    else if constexpr (OPERATION == SfpuTernaryOp::lerp)
    {
        SfpuTernaryFn<sfpu::calculate_lerp<APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, vector_mode);
    }
    else if constexpr (OPERATION == SfpuTernaryOp::snake_beta)
    {
        SfpuTernaryFn<sfpu::calculate_snake_beta<APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, ITERATIONS>, DST_SYNC_MODE, DST_ACCUM_MODE>::calculate(
            dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, vector_mode);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported ternary operation");
    }
}

} // namespace test_utils
