// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "quasar_sfpu_test_operations.h"

// To add a new Quasar unary SFPU operation:
// 1. Include its `ckernel_sfpu_<op>.h` below.
// 2. Add the `SfpuType` enumerator to the `if constexpr` chain in
//    call_unary_sfpu_operation_quasar() (and to init_unary_sfpu_operation_quasar()
//    if the op needs an init step).
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_log.h"
#include "llk_sfpu/ckernel_sfpu_log1p.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"
#include "llk_sfpu/ckernel_sfpu_rsqrt.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_sqrt_custom.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_trigonometry.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"

// Binary SFPU op headers (consumed by the binary dispatchers below). The op is
// selected via the LLK ckernel::BinaryOp enum (reused like Blackhole; the
// comparison and max/min enumerators were added to it in llk_defs.h).
//
// To add a new Quasar binary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the enumerator to ckernel::BinaryOp (llk_defs.h) if it is not there.
// 3. Add the `if constexpr` branch in call_binary_sfpu_operation_quasar()
//    (and init_binary_sfpu_operation_quasar() if it needs an init step).
#include "llk_sfpu/ckernel_sfpu_binary.h"         // calculate_sfpu_binary / sfpu_binary_init (float mul/div)
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h" // calculate_binary_max_min / _init_binary_max_min_
#include "llk_sfpu/ckernel_sfpu_quant.h"          // quant_family / quant_family_init (quant/requant/dequant)
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_add.h"         // _add_int_ (int add)
#include "sfpu/ckernel_sfpu_binary_comp.h" // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"   // _mul_int32_ (int mul)

namespace ckernel::sfpu
{
// Test-only tile-loop adapter for the production per-vector sqrt_custom primitive.
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_sqrt_custom()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::dst_reg[0] = sfpu_sqrt_custom<APPROXIMATION_MODE>(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}
} // namespace ckernel::sfpu

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

template <auto>
inline constexpr bool unhandled_op = false;

/**
 * @brief Whether OPERATION is one of the six comparison-to-zero modes.
 *
 * The comp family needs a runtime format switch (@ref call_zero_comp_operation_quasar)
 * to pick the integer-vs-float compare path, unlike the float-only unary ops, so the
 * dispatcher special-cases it.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_zero_comp_op(QuasarSfpuTestOperation op)
{
    return op == QuasarSfpuTestOperation::equal_zero || op == QuasarSfpuTestOperation::not_equal_zero || op == QuasarSfpuTestOperation::less_than_zero ||
           op == QuasarSfpuTestOperation::greater_than_zero || op == QuasarSfpuTestOperation::less_than_equal_zero ||
           op == QuasarSfpuTestOperation::greater_than_equal_zero;
}

template <QuasarSfpuTestOperation OPERATION>
inline constexpr SfpuType zero_comp_kernel_operation()
{
    static_assert(is_zero_comp_op(OPERATION));
    if constexpr (OPERATION == QuasarSfpuTestOperation::equal_zero)
    {
        return SfpuType::equal_zero;
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::not_equal_zero)
    {
        return SfpuType::not_equal_zero;
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::less_than_zero)
    {
        return SfpuType::less_than_zero;
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::greater_than_zero)
    {
        return SfpuType::greater_than_zero;
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::less_than_equal_zero)
    {
        return SfpuType::less_than_equal_zero;
    }
    else
    {
        return SfpuType::greater_than_equal_zero;
    }
}

/**
 * @brief Whether OPERATION is one of the trigonometry / inverse-hyperbolic ops.
 *
 * They share one init (@ref init_trigonometry, which programs ADDR_MOD_6 for the
 * auto-incrementing Dest store) since every trig body has the same load/compute/store shape.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_trig_op(QuasarSfpuTestOperation op)
{
    return op == QuasarSfpuTestOperation::sine || op == QuasarSfpuTestOperation::cosine || op == QuasarSfpuTestOperation::tan ||
           op == QuasarSfpuTestOperation::atan || op == QuasarSfpuTestOperation::asin || op == QuasarSfpuTestOperation::acos ||
           op == QuasarSfpuTestOperation::sinh || op == QuasarSfpuTestOperation::cosh || op == QuasarSfpuTestOperation::acosh ||
           op == QuasarSfpuTestOperation::asinh || op == QuasarSfpuTestOperation::atanh;
}

/**
 * @brief Run the per-operation init step for a Quasar unary SFPU op.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @note Pair with @ref call_unary_sfpu_operation_quasar for the calculate step.
 */
template <QuasarSfpuTestOperation OPERATION, bool is_fp32_dest_acc_en, bool APPROX = false>
void init_unary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == QuasarSfpuTestOperation::gelu)
    {
        gelu_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::square)
    {
        init_square();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::rsqrt)
    {
        _init_rsqrt_<APPROX>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::reciprocal)
    {
        _init_reciprocal_<APPROX>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::log)
    {
        _init_log_<APPROX>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::log1p)
    {
        log1p_init<APPROX, false /* FAST_APPROX */, is_fp32_dest_acc_en>();
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        init_zero_comp();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::typecast)
    {
        init_typecast();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sine)
    {
        sine_init<APPROX>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::cosine)
    {
        cosine_init<APPROX>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::tan)
    {
        tangent_init<APPROX>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::atan)
    {
        atan_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::asin || OPERATION == QuasarSfpuTestOperation::acos)
    {
        math::_reset_counters_<p_setrwc::SET_ABD_F>();
        asin_acos_init<is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sinh)
    {
        sinh_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::cosh)
    {
        cosh_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::acosh || OPERATION == QuasarSfpuTestOperation::asinh)
    {
        init_inverse_hyperbolic<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::atanh)
    {
        init_atanh<APPROX, is_fp32_dest_acc_en>();
    }
}

/**
 * @brief Apply a comparison-to-zero SFPU op in-place on one Dest tile.
 *
 * Unlike the float-only unary ops, comp needs the SFPU math format at runtime to
 * pick the integer load/store width and the integer-vs-float compare path (see
 * `ckernel_sfpu_comp.h`). Int32/Int16/Int8/UInt16/UInt8 select their explicit
 * sfpmem width; all float widths share the width-agnostic `Float32` instantiation,
 * whose sfpi compare path resolves the actual width from the HW format config.
 *
 * @tparam OPERATION The comparison-to-zero `SfpuType` (compile-time constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format selecting the sfpmem mode / result encoding.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <QuasarSfpuTestOperation OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_zero_comp_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format)
{
    static_assert(is_zero_comp_op(OPERATION), "call_zero_comp_operation_quasar: OPERATION must be a comparison-to-zero SfpuType");
    constexpr SfpuType kernel_operation = zero_comp_kernel_operation<OPERATION>();

    switch (sfpu_format)
    {
        case DataFormat::Int32:
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Int32, kernel_operation, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::Int16:
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Int16, kernel_operation, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::Int8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::Int8;
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, sfpu_fmt, kernel_operation, ITERATIONS), dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::UInt16:
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::UInt16, kernel_operation, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::UInt8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::UInt8;
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, sfpu_fmt, kernel_operation, ITERATIONS), dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::Float32:
            // Float widths share the width-agnostic Float32 path: its sfpmem::DEFAULT access mode
            // resolves the actual width from ALU_FORMAT_SPEC_REG / ACC_CTRL.
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Float32, kernel_operation, ITERATIONS), dst_index, VectorMode::RC);
            break;
        default:
            LLK_ASSERT(false, "Unsupported Quasar comp-to-zero SFPU format");
            break;
    }
}

/**
 * @brief Apply a Quasar unary SFPU op in-place on one Dest tile.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam APPROX Whether operations with approximate and accurate paths use the approximate path.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @tparam TYPECAST_IN_FORMAT Source format for the typecast op (default Float32).
 * @tparam TYPECAST_OUT_FORMAT Destination format for the typecast op (default Float16_b).
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format; only the comp family reads it (see
 *        @ref call_zero_comp_operation_quasar), float-only ops ignore it.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <
    QuasarSfpuTestOperation OPERATION,
    DstSync DST_SYNC,
    bool is_fp32_dest_acc_en,
    bool APPROX                    = false,
    int ITERATIONS                 = SFPU_ITERATIONS,
    DataFormat TYPECAST_IN_FORMAT  = DataFormat::Float32,
    DataFormat TYPECAST_OUT_FORMAT = DataFormat::Float16_b>
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32, VectorMode vector_mode = VectorMode::RC)
{
    if constexpr (OPERATION == QuasarSfpuTestOperation::abs)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_abs_, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::exponential)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_exponential,
            (APPROX, is_fp32_dest_acc_en, false, ITERATIONS),
            dst_index,
            vector_mode,
            p_sfpu::kCONST_1_FP16B);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::gelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::relu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_relu_, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::reciprocal)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_reciprocal, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::log)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            _calculate_log_,
            (APPROX, false /* HAS_BASE_SCALING */, ITERATIONS),
            dst_index,
            vector_mode,
            ITERATIONS,
            0u /* log_base_scale_factor */);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::log1p)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_log1p, (APPROX, false /* FAST_APPROX */, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sqrt_, (true /* APPROX */, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sqrt_custom)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sqrt_custom, (APPROX, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::tanh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sigmoid)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sigmoid_, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::silu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_silu_, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::rsqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rsqrt, (APPROX, ITERATIONS, is_fp32_dest_acc_en), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::square)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_square, (ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sine)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sine, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::cosine)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cosine, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::tan)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tangent, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::atan)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_atan, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::asin)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_asin, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::acos)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_acos, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::sinh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sinh, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::cosh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cosh, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::acosh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_acosh, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::asinh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_asinh, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::atanh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_atanh, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::negative)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_negative_, (false, ITERATIONS), dst_index, vector_mode);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::softplus)
    {
        // Softplus params beta / (1/beta) / threshold as fp32 bit patterns, matching the
        // UnarySFPUGolden._softplus reference defaults (beta = 1.0, threshold = 20.0).
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_softplus,
            (false, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            vector_mode,
            static_cast<std::uint32_t>(0x3F800000),  // beta = 1.0 (fp32)
            static_cast<std::uint32_t>(0x3F800000),  // 1/beta = 1.0 (fp32)
            static_cast<std::uint32_t>(0x41A00000)); // threshold = 20.0 (fp32)
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::clamp)
    {
        // Clamp bounds fixed to [-1.0, +1.0] as fp32 bit patterns (matching the UnarySFPUGolden._clamp
        // reference). Extra args are forwarded to the per-face functor call.
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_clamp,
            (false, ITERATIONS),
            dst_index,
            vector_mode,
            static_cast<std::uint32_t>(0xBF800000),  // min = -1.0 (fp32)
            static_cast<std::uint32_t>(0x3F800000)); // max = +1.0 (fp32)
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        call_zero_comp_operation_quasar<OPERATION, DST_SYNC, is_fp32_dest_acc_en, ITERATIONS>(dst_index, sfpu_format);
    }
    else if constexpr (OPERATION == QuasarSfpuTestOperation::typecast)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_typecast, (TYPECAST_IN_FORMAT, TYPECAST_OUT_FORMAT, ITERATIONS), dst_index, vector_mode);
    }
    else
    {
        static_assert(unhandled_op<OPERATION>, "call_unary_sfpu_operation_quasar: unhandled Quasar unary SFPU operation");
    }
}

constexpr bool quasar_binary_op_is_max_min(ckernel::BinaryOp op)
{
    return op == ckernel::BinaryOp::MAX || op == ckernel::BinaryOp::MIN;
}

constexpr bool quasar_binary_op_is_quant(ckernel::BinaryOp op)
{
    return op == ckernel::BinaryOp::QUANT || op == ckernel::BinaryOp::REQUANT || op == ckernel::BinaryOp::DEQUANT;
}

// Map the shared BinaryOp enum onto the quant kernel's op-templated QuantVariant.
template <ckernel::BinaryOp OP>
constexpr ckernel::sfpu::QuantVariant quant_variant_of()
{
    if constexpr (OP == ckernel::BinaryOp::QUANT)
    {
        return ckernel::sfpu::QuantVariant::Quant;
    }
    else if constexpr (OP == ckernel::BinaryOp::REQUANT)
    {
        return ckernel::sfpu::QuantVariant::Requant;
    }
    else if constexpr (OP == ckernel::BinaryOp::DEQUANT)
    {
        return ckernel::sfpu::QuantVariant::Dequant;
    }
    else
    {
        static_assert(unhandled_op<OP>, "quant_variant_of: unhandled quant BinaryOp");
    }
}

/**
 * @brief Run the per-operation init step for a Quasar binary SFPU op.
 *
 * @tparam OP The binary op (compile-time `ckernel::BinaryOp` constant).
 * @tparam SIGN_MAGNITUDE_FORMAT Quant family only: if true, treat int32 Dest as SMAG32
 *         and skip the sign-magnitude<->2's-complement casts. Must match the calculate step.
 * @param zero_point fp32 bit-pattern of the zero-point loaded once by the quant
 *        family init (DEQUANT expects the bits of -zero_point); ignored by the
 *        other ops, which have no runtime init argument.
 * @note Pair with @ref call_binary_sfpu_operation_quasar for the calculate step.
 */
template <ckernel::BinaryOp OP, bool SIGN_MAGNITUDE_FORMAT = false>
void init_binary_sfpu_operation_quasar([[maybe_unused]] std::uint32_t zero_point = 0)
{
    if constexpr (OP == BinaryOp::MUL)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::MUL>(); // no-op for MUL; harmless on the int path
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        sfpu_binary_init<false /*APPROX*/, BinaryOp::DIV>();
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        _init_binary_max_min_();
    }
    else if constexpr (quasar_binary_op_is_quant(OP))
    {
        // One op-templated quant kernel; DEQUANT's caller passes bits of -zero_point.
        quant_family_init<quant_variant_of<OP>(), SIGN_MAGNITUDE_FORMAT>(zero_point);
    }
    // ADD / SUB / GT / LT / LE / GE are stateless — no init.
}

/**
 * @brief Apply a Quasar binary SFPU op over two Dest operands into a result tile.
 *
 * @tparam OP The binary op (compile-time `ckernel::BinaryOp` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam dst_rounding_mode Controls bf16 narrowing for ADD/SUB results. Default truncates;
 *         NearestEven applies software RNE before the store. Ignored for MUL (no narrowing)
 *         and DIV (always rounds RNE regardless). No-op when is_fp32_dest_acc_en is true.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @tparam SIGN_MAGNITUDE_FORMAT Quant family only: if true, treat int32 Dest as SMAG32
 *         and skip the sign-magnitude<->2's-complement casts. Must match the init step.
 * @param src0_tile,src1_tile,dst_tile Operand / result tile indices.
 * @param math_format Dest math format (Int32 vs float path for MUL and max/min).
 * @note Must be preceded by @ref init_binary_sfpu_operation_quasar for the same op.
 */
template <
    ckernel::BinaryOp OP,
    DstSync DST_SYNC,
    bool is_fp32_dest_acc_en,
    ckernel::DstRoundingMode dst_rounding_mode = ckernel::DstRoundingMode::Default,
    int ITERATIONS                             = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT                 = false>
void call_binary_sfpu_operation_quasar(std::uint32_t src0_tile, std::uint32_t src1_tile, std::uint32_t dst_tile, [[maybe_unused]] DataFormat math_format)
{
    if constexpr (OP == BinaryOp::ADD)
    {
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, _add_int_, (false, ITERATIONS, DataFormat::Int32, 0, false), src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_sfpu_binary,
                (false /*APPROX*/, BinaryOp::ADD, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::SUB)
    {
        // Int32 SUB is not ported to Quasar (sub_int_sfpu.h is WH-only); float path only.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary,
            (false /*APPROX*/, BinaryOp::SUB, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::GT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::gt), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::lt), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LE)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::le), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::GE)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_binary_comp_int32, (false, ITERATIONS, SfpuType::ge), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::MUL)
    {
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _mul_int32_, (false, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_sfpu_binary,
                (false /*APPROX*/, BinaryOp::MUL, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary,
            (false /*APPROX*/, BinaryOp::DIV, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (quasar_binary_op_is_quant(OP))
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            quant_family,
            (quant_variant_of<OP>(), ITERATIONS, SIGN_MAGNITUDE_FORMAT),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        constexpr bool IS_MAX = (OP == BinaryOp::MAX);
        // All integer formats route through the Int32 path; float / MX use Float32.
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_binary_max_min,
                (DataFormat::Int32, IS_MAX, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_binary_max_min,
                (DataFormat::Float32, IS_MAX, ITERATIONS),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    else
    {
        static_assert(unhandled_op<OP>, "call_binary_sfpu_operation_quasar: unhandled Quasar binary SFPU operation");
    }
}

} // namespace test_utils
