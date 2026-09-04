// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_sfpu/llk_math_eltwise_sfpu_op.h"
#include "sfpu_test_ops.h"

// To add a new Quasar unary SFPU operation:
// 1. Include its `ckernel_sfpu_<op>.h` below.
// 2. Add the enumerator to SfpuUnaryOp in sfpu_test_ops.h and the `if constexpr`
//    chain in call_unary_sfpu_operation_quasar() (and to init_unary_sfpu_operation_quasar()
//    if the op needs an init step).
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_cumsum.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"
#include "llk_sfpu/ckernel_sfpu_rsqrt.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
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
// comparison, max/min, and atan2 enumerators were added to it in ckernel_defs.h).
//
// To add a new Quasar binary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the enumerator to ckernel::BinaryOp (tt_llk_quasar/common/inc/ckernel_defs.h) if it is not there.
// 3. Add the `if constexpr` branch in call_binary_sfpu_operation_quasar()
//    (and init_binary_sfpu_operation_quasar() if it needs an init step).
#include "llk_sfpu/ckernel_sfpu_atan2.h"          // calculate_sfpu_atan2 / calculate_sfpu_atan2_init (float atan2)
#include "llk_sfpu/ckernel_sfpu_binary.h"         // calculate_sfpu_binary / sfpu_binary_init (float mul/div)
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h" // calculate_binary_max_min / _init_binary_max_min_
#include "llk_sfpu/ckernel_sfpu_quant.h"          // quant_family / quant_family_init (quant/requant/dequant)
#include "sfpu/ckernel_sfpu_add.h"                // _add_int_ (int add)
#include "sfpu/ckernel_sfpu_binary_comp.h"        // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"          // _mul_int32_ (int mul)

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

template <auto>
inline constexpr bool unhandled_op = false;

template <SfpuUnaryOp OPERATION>
constexpr ZeroCompMode zero_comp_mode_of_q()
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
constexpr TrigOp trig_op_of_q()
{
    if constexpr (OPERATION == SfpuUnaryOp::sine)
    {
        return TrigOp::Sine;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cosine)
    {
        return TrigOp::Cosine;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::acosh)
    {
        return TrigOp::Acosh;
    }
    else if constexpr (OPERATION == SfpuUnaryOp::asinh)
    {
        return TrigOp::Asinh;
    }
    else
    {
        return TrigOp::Atanh;
    }
}

/**
 * @brief Whether OPERATION is one of the six comparison-to-zero modes.
 *
 * The comp family needs a runtime format switch (@ref call_zero_comp_operation_quasar)
 * to pick the integer-vs-float compare path, unlike the float-only unary ops, so the
 * dispatcher special-cases it.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_zero_comp_op(SfpuUnaryOp op)
{
    return op == SfpuUnaryOp::equal_zero || op == SfpuUnaryOp::not_equal_zero || op == SfpuUnaryOp::less_than_zero || op == SfpuUnaryOp::greater_than_zero ||
           op == SfpuUnaryOp::less_than_equal_zero || op == SfpuUnaryOp::greater_than_equal_zero;
}

/**
 * @brief Whether OPERATION is one of the trigonometry / inverse-hyperbolic ops.
 *
 * They share one init (@ref init_trigonometry, which programs ADDR_MOD_6 for the
 * auto-incrementing Dest store) since every trig body has the same load/compute/store shape.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_trig_op(SfpuUnaryOp op)
{
    return op == SfpuUnaryOp::sine || op == SfpuUnaryOp::cosine || op == SfpuUnaryOp::acosh || op == SfpuUnaryOp::asinh || op == SfpuUnaryOp::atanh;
}

/**
 * @brief Run the per-operation init step for a Quasar unary SFPU op.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuUnaryOp` constant).
 * @note Pair with @ref call_unary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuUnaryOp OPERATION, bool is_fp32_dest_acc_en, bool APPROX = false>
void init_unary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuUnaryOp::gelu)
    {
        gelu_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::square)
    {
        init_square();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rsqrt)
    {
        _init_rsqrt_<APPROX>();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::reciprocal)
    {
        _init_reciprocal_<APPROX>();
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        init_zero_comp();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::typecast)
    {
        init_typecast();
    }
    else if constexpr (is_trig_op(OPERATION))
    {
        init_trigonometry<trig_op_of_q<OPERATION>(), is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cumsum)
    {
        cumsum_init<APPROX>();
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
 * @tparam OPERATION The comparison-to-zero `SfpuUnaryOp` (compile-time constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format selecting the sfpmem mode / result encoding.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <SfpuUnaryOp OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_zero_comp_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format)
{
    static_assert(is_zero_comp_op(OPERATION), "call_zero_comp_operation_quasar: OPERATION must be a comparison-to-zero SfpuUnaryOp");

    switch (sfpu_format)
    {
        case DataFormat::Int32:
            SfpuUnaryFn<sfpu::calculate_zero_comp<false, DataFormat::Int32, zero_comp_mode_of_q<OPERATION>(), ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::
                calculate(dst_index, VectorMode::RC);
            break;
        case DataFormat::Int16:
            SfpuUnaryFn<sfpu::calculate_zero_comp<false, DataFormat::Int16, zero_comp_mode_of_q<OPERATION>(), ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::
                calculate(dst_index, VectorMode::RC);
            break;
        case DataFormat::Int8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::Int8;
            SfpuUnaryFn<sfpu::calculate_zero_comp<false, sfpu_fmt, zero_comp_mode_of_q<OPERATION>(), ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
                dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::UInt16:
            SfpuUnaryFn<sfpu::calculate_zero_comp<false, DataFormat::UInt16, zero_comp_mode_of_q<OPERATION>(), ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::
                calculate(dst_index, VectorMode::RC);
            break;
        case DataFormat::UInt8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::UInt8;
            SfpuUnaryFn<sfpu::calculate_zero_comp<false, sfpu_fmt, zero_comp_mode_of_q<OPERATION>(), ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
                dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::Float32:
            // Float widths share the width-agnostic Float32 path: its sfpmem::DEFAULT access mode
            // resolves the actual width from ALU_FORMAT_SPEC_REG / ACC_CTRL.
            SfpuUnaryFn<sfpu::calculate_zero_comp<false, DataFormat::Float32, zero_comp_mode_of_q<OPERATION>(), ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::
                calculate(dst_index, VectorMode::RC);
            break;
        default:
            LLK_ASSERT(false, "Unsupported Quasar comp-to-zero SFPU format");
            break;
    }
}

/**
 * @brief Apply a Quasar unary SFPU op in-place on one Dest tile.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuUnaryOp` constant).
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam APPROX Whether operations with approximate and accurate paths use the approximate path.
 * @tparam ITERATIONS Number of SFPU loop iterations.
 * @tparam TYPECAST_IN_FORMAT Source format for the typecast op (default Float32).
 * @tparam TYPECAST_OUT_FORMAT Destination format for the typecast op (default Float16_b).
 * @param dst_index Destination tile index operated on (already offset by DST_INDEX).
 * @param sfpu_format SFPU math format; only the comp family reads it (see
 *        @ref call_zero_comp_operation_quasar), float-only ops ignore it.
 * @param first Whether this tile starts a fresh top-to-bottom accumulation chain; only cumsum
 *        reads it. Defaults to true so each tile is independent.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <
    SfpuUnaryOp OPERATION,
    DstSync DST_SYNC,
    bool is_fp32_dest_acc_en,
    bool APPROX                    = false,
    int ITERATIONS                 = SFPU_ITERATIONS,
    DataFormat TYPECAST_IN_FORMAT  = DataFormat::Float32,
    DataFormat TYPECAST_OUT_FORMAT = DataFormat::Float16_b>
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32, [[maybe_unused]] const bool first = true)
{
    if constexpr (OPERATION == SfpuUnaryOp::abs)
    {
        SfpuUnaryFn<sfpu::_calculate_abs_<ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::exponential)
    {
        SfpuUnaryFn<sfpu::calculate_exponential<APPROX, is_fp32_dest_acc_en, false, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            dst_index, VectorMode::RC, p_sfpu::kCONST_1_FP16B);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::gelu)
    {
        SfpuUnaryFn<sfpu::calculate_gelu<APPROX, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::relu)
    {
        SfpuUnaryFn<sfpu::_calculate_relu_<ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::reciprocal)
    {
        SfpuUnaryFn<sfpu::calculate_reciprocal<APPROX, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sqrt)
    {
        SfpuUnaryFn<sfpu::_calculate_sqrt_<true /* APPROX */, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::tanh)
    {
        SfpuUnaryFn<sfpu::calculate_tanh<ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::sigmoid)
    {
        SfpuUnaryFn<sfpu::_calculate_sigmoid_<ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::silu)
    {
        SfpuUnaryFn<sfpu::_calculate_silu_<ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::rsqrt)
    {
        SfpuUnaryFn<sfpu::calculate_rsqrt<APPROX, ITERATIONS, is_fp32_dest_acc_en>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::square)
    {
        SfpuUnaryFn<sfpu::calculate_square<ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (is_trig_op(OPERATION))
    {
        // One op-templated kernel serves sine/cosine/acosh/asinh/atanh; OPERATION picks the branch
        // at compile time. APPROXIMATION_MODE=false selects the full-polynomial (accurate) path.
        SfpuUnaryFn<
            sfpu::calculate_trigonometry<trig_op_of_q<OPERATION>(), false /* APPROX */, is_fp32_dest_acc_en, ITERATIONS>,
            DST_SYNC,
            is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::negative)
    {
        SfpuUnaryFn<sfpu::_calculate_negative_<false, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::softplus)
    {
        // Softplus params beta / (1/beta) / threshold as fp32 bit patterns, matching the
        // UnarySFPUGolden._softplus reference defaults (beta = 1.0, threshold = 20.0).
        SfpuUnaryFn<sfpu::calculate_softplus<false, is_fp32_dest_acc_en, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            dst_index,
            VectorMode::RC,
            static_cast<std::uint32_t>(0x3F800000),
            /* beta = 1.0 (fp32) */ static_cast<std::uint32_t>(0x3F800000),
            /* 1/beta = 1.0 (fp32) */ static_cast<std::uint32_t>(0x41A00000)); // threshold = 20.0 (fp32)
    }
    else if constexpr (OPERATION == SfpuUnaryOp::clamp)
    {
        // Clamp bounds fixed to [-1.0, +1.0] as fp32 bit patterns (matching the UnarySFPUGolden._clamp
        // reference). Extra args are forwarded to the per-face functor call.
        SfpuUnaryFn<sfpu::calculate_clamp<false, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            dst_index,
            VectorMode::RC,
            static_cast<std::uint32_t>(0xBF800000),
            /* min = -1.0 (fp32) */ static_cast<std::uint32_t>(0x3F800000)); // max = +1.0 (fp32)
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        call_zero_comp_operation_quasar<OPERATION, DST_SYNC, is_fp32_dest_acc_en, ITERATIONS>(dst_index, sfpu_format);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::typecast)
    {
        SfpuUnaryFn<sfpu::calculate_typecast<TYPECAST_IN_FORMAT, TYPECAST_OUT_FORMAT, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuUnaryOp::cumsum)
    {
        // Whole-tile op: the accumulation chain spans all 32 tile rows and crosses the face-pair
        // boundary, so it runs once per tile (RC_custom), not once per face.
        SfpuUnaryFn<sfpu::calculate_cumsum<APPROX, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(dst_index, VectorMode::RC_custom, first);
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
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode. Must match the calculate step;
 *         atan2 uses it to select the reciprocal variant its polynomial expects.
 * @tparam SIGN_MAGNITUDE_FORMAT Quant family only: if true, treat int32 Dest as SMAG32
 *         and skip the sign-magnitude<->2's-complement casts. Must match the calculate step.
 * @tparam APPROXIMATION_MODE Whether to use the operation's approximate path. Must match the
 *         calculate step; atan2 uses it to select the LUT-only reciprocal path.
 * @param zero_point fp32 bit-pattern of the zero-point loaded once by the quant
 *        family init (DEQUANT expects the bits of -zero_point); ignored by the
 *        other ops, which have no runtime init argument.
 * @note Pair with @ref call_binary_sfpu_operation_quasar for the calculate step.
 */
template <ckernel::BinaryOp OP, bool is_fp32_dest_acc_en = false, bool SIGN_MAGNITUDE_FORMAT = false, bool APPROXIMATION_MODE = false>
void init_binary_sfpu_operation_quasar([[maybe_unused]] std::uint32_t zero_point = 0)
{
    if constexpr (OP == BinaryOp::MUL)
    {
        sfpu_binary_init<APPROXIMATION_MODE, BinaryOp::MUL>(); // no-op for MUL; harmless on the int path
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        // Forwards APPROXIMATION_MODE to _init_reciprocal_ (LUT-only vs Newton).
        sfpu_binary_init<APPROXIMATION_MODE, BinaryOp::DIV>();
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
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        // Programs the Newton-Raphson reciprocal constant. is_fp32_dest_acc_en must be the
        // same value the calculate step uses — it picks both the minimax degree and the
        // reciprocal variant.
        calculate_sfpu_atan2_init<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
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
 * @tparam APPROXIMATION_MODE Whether to use the operation's approximate path. Must match the
 *         init step; atan2 uses it to select the LUT-only reciprocal path.
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
    bool SIGN_MAGNITUDE_FORMAT                 = false,
    bool APPROXIMATION_MODE                    = false>
void call_binary_sfpu_operation_quasar(std::uint32_t src0_tile, std::uint32_t src1_tile, std::uint32_t dst_tile, [[maybe_unused]] DataFormat math_format)
{
    if constexpr (OP == BinaryOp::ADD)
    {
        if (math_format == DataFormat::Int32)
        {
            SfpuBinaryFn<sfpu::_add_int_<false, ITERATIONS, DataFormat::Int32, 0, false>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
                src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SfpuBinaryFn<
                sfpu::calculate_sfpu_binary<APPROXIMATION_MODE, BinaryOp::ADD, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS>,
                DST_SYNC,
                is_fp32_dest_acc_en>::calculate(src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::SUB)
    {
        // Int32 SUB is not ported to Quasar (sub_int_sfpu.h is WH-only); float path only.
        SfpuBinaryFn<
            sfpu::calculate_sfpu_binary<APPROXIMATION_MODE, BinaryOp::SUB, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS>,
            DST_SYNC,
            is_fp32_dest_acc_en>::calculate(src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::GT)
    {
        SfpuBinaryFn<sfpu::calculate_binary_comp_int32<false, ITERATIONS, BinaryCompMode::Gt>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LT)
    {
        SfpuBinaryFn<sfpu::calculate_binary_comp_int32<false, ITERATIONS, BinaryCompMode::Lt>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LE)
    {
        SfpuBinaryFn<sfpu::calculate_binary_comp_int32<false, ITERATIONS, BinaryCompMode::Le>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::GE)
    {
        SfpuBinaryFn<sfpu::calculate_binary_comp_int32<false, ITERATIONS, BinaryCompMode::Ge>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::MUL)
    {
        if (math_format == DataFormat::Int32)
        {
            SfpuBinaryFn<sfpu::_mul_int32_<false, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SfpuBinaryFn<
                sfpu::calculate_sfpu_binary<APPROXIMATION_MODE, BinaryOp::MUL, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS>,
                DST_SYNC,
                is_fp32_dest_acc_en>::calculate(src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::DIV)
    {
        SfpuBinaryFn<
            sfpu::calculate_sfpu_binary<APPROXIMATION_MODE, BinaryOp::DIV, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS>,
            DST_SYNC,
            is_fp32_dest_acc_en>::calculate(src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        // atan2(y, x): src0 = y, src1 = x. is_fp32_dest_acc_en must match the init's.
        SfpuBinaryFn<sfpu::calculate_sfpu_atan2<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (quasar_binary_op_is_quant(OP))
    {
        SfpuBinaryFn<sfpu::quant_family<quant_variant_of<OP>(), ITERATIONS, SIGN_MAGNITUDE_FORMAT>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
            src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (quasar_binary_op_is_max_min(OP))
    {
        constexpr bool IS_MAX = (OP == BinaryOp::MAX);
        // All integer formats route through the Int32 path; float / MX use Float32.
        if (math_format == DataFormat::Int32)
        {
            SfpuBinaryFn<sfpu::calculate_binary_max_min<DataFormat::Int32, IS_MAX, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
                src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
        else
        {
            SfpuBinaryFn<sfpu::calculate_binary_max_min<DataFormat::Float32, IS_MAX, ITERATIONS>, DST_SYNC, is_fp32_dest_acc_en>::calculate(
                src0_tile, src1_tile, dst_tile, VectorMode::RC);
        }
    }
    else
    {
        static_assert(unhandled_op<OP>, "call_binary_sfpu_operation_quasar: unhandled Quasar binary SFPU operation");
    }
}

} // namespace test_utils
