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
#include "llk_sfpu/ckernel_sfpu_activations.h"
#include "llk_sfpu/ckernel_sfpu_addcdiv.h"
#include "llk_sfpu/ckernel_sfpu_addcmul.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_binary_comp.h"
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
#include "llk_sfpu/ckernel_sfpu_signbit.h"
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
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_hardtanh.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_negative.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sub_int.h"
#include "sfpu/ckernel_sfpu_threshold.h"

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
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(exp2_init<APPROX_MODE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(exp_init<APPROX_MODE, 0x3F800000 /* exp_base_scale_factor */, CLAMP_NEGATIVE, is_fp32_dest_acc_en>);
    }
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(erfinv_init<APPROX_MODE>);
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
    else if constexpr (OPERATION == SfpuType::signbit)
    {
        llk_math_eltwise_unary_sfpu_init<OPERATION>(signbit_init);
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
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_erfinv, (APPROX_MODE), dst_index, VectorMode::RC);
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
 */
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 32, std::uint32_t MATH_FORMAT = 0>
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
    else
    {
        // BinaryOps without a dedicated SfpuType use the baseline binary addrmod setup.
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
                (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32, false /* SIGN_MAGNITUDE_FORMAT */),
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
                (APPROXIMATION_MODE, PER_FACE_ITERATIONS, ckernel::InstrModLoadStore::INT32, false /* SIGN_MAGNITUDE_FORMAT */),
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
