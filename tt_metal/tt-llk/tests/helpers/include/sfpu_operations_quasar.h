// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

// To add a new Quasar unary SFPU operation:
// 1. Include its `ckernel_sfpu_<op>.h` below.
// 2. Add the `SfpuType` enumerator to the `if constexpr` chain in
//    call_unary_sfpu_operation_quasar() (and to init_unary_sfpu_operation_quasar()
//    if the op needs an init step).
#include "llk_sfpu/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_activations.h"
#include "llk_sfpu/ckernel_sfpu_add1.h"
#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_bitwise_not.h"
#include "llk_sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h"
#include "llk_sfpu/ckernel_sfpu_cbrt.h"
#include "llk_sfpu/ckernel_sfpu_celu.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_cumsum.h"
#include "llk_sfpu/ckernel_sfpu_digamma.h"
#include "llk_sfpu/ckernel_sfpu_elu.h"
#include "llk_sfpu/ckernel_sfpu_erf.h"
#include "llk_sfpu/ckernel_sfpu_erfc.h"
#include "llk_sfpu/ckernel_sfpu_erfinv.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_exp2.h"
#include "llk_sfpu/ckernel_sfpu_expm1.h"
#include "llk_sfpu/ckernel_sfpu_fmod.h"
#include "llk_sfpu/ckernel_sfpu_gelu.h"
#include "llk_sfpu/ckernel_sfpu_hardmish.h"
#include "llk_sfpu/ckernel_sfpu_hardshrink.h"
#include "llk_sfpu/ckernel_sfpu_hardtanh.h"
#include "llk_sfpu/ckernel_sfpu_heaviside.h"
#include "llk_sfpu/ckernel_sfpu_i0.h"
#include "llk_sfpu/ckernel_sfpu_i1.h"
#include "llk_sfpu/ckernel_sfpu_identity.h"
#include "llk_sfpu/ckernel_sfpu_isinf_isnan.h"
#include "llk_sfpu/ckernel_sfpu_lgamma.h"
#include "llk_sfpu/ckernel_sfpu_logical_not.h"
#include "llk_sfpu/ckernel_sfpu_mish.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "llk_sfpu/ckernel_sfpu_polygamma.h"
#include "llk_sfpu/ckernel_sfpu_prelu.h"
#include "llk_sfpu/ckernel_sfpu_rdiv.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"
#include "llk_sfpu/ckernel_sfpu_relu.h"
#include "llk_sfpu/ckernel_sfpu_remainder.h"
#include "llk_sfpu/ckernel_sfpu_rounding_ops.h"
#include "llk_sfpu/ckernel_sfpu_rpow.h"
#include "llk_sfpu/ckernel_sfpu_rsqrt.h"
#include "llk_sfpu/ckernel_sfpu_rsub_int32.h"
#include "llk_sfpu/ckernel_sfpu_selu.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid_appx.h"
#include "llk_sfpu/ckernel_sfpu_sign.h"
#include "llk_sfpu/ckernel_sfpu_softcap.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_softshrink.h"
#include "llk_sfpu/ckernel_sfpu_softsign.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_tanh_derivative.h"
#include "llk_sfpu/ckernel_sfpu_tanhshrink.h"
#include "llk_sfpu/ckernel_sfpu_threshold.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"
#include "llk_sfpu/ckernel_sfpu_trigonometry.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "llk_sfpu/ckernel_sfpu_unary_comp.h"
#include "llk_sfpu/ckernel_sfpu_unary_power.h"
#include "llk_sfpu/ckernel_sfpu_unary_shift.h"
#include "llk_sfpu/ckernel_sfpu_xielu.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_typecast_fp32_to_uint16.h"

// Binary SFPU op headers (consumed by the binary dispatchers below). The op is
// selected via the LLK ckernel::BinaryOp enum (reused like Blackhole; the
// comparison, max/min, and atan2 enumerators were added to it in ckernel_defs.h).
//
// To add a new Quasar binary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the enumerator to ckernel::BinaryOp (tt_llk_quasar/common/inc/ckernel_defs.h) if it is not there.
// 3. Add the `if constexpr` branch in call_binary_sfpu_operation_quasar()
//    (and init_binary_sfpu_operation_quasar() if it needs an init step).
#include "llk_sfpu/ckernel_sfpu_add.h"              // calculate_add_int (int add)
#include "llk_sfpu/ckernel_sfpu_add_top_row.h"      // calculate_add_top_row (top four rows of two tiles, Float32/Int32)
#include "llk_sfpu/ckernel_sfpu_atan2.h"            // calculate_sfpu_atan2 / calculate_sfpu_atan2_init (float atan2)
#include "llk_sfpu/ckernel_sfpu_binary.h"           // calculate_sfpu_binary / sfpu_binary_init (float mul/div)
#include "llk_sfpu/ckernel_sfpu_binary_bitwise.h"   // calculate_sfpu_binary_bitwise (int32 and/or/xor)
#include "llk_sfpu/ckernel_sfpu_binary_fmod.h"      // calculate_sfpu_binary_fmod / calculate_fmod_int32
#include "llk_sfpu/ckernel_sfpu_binary_max_min.h"   // calculate_binary_max_min / _init_binary_max_min_
#include "llk_sfpu/ckernel_sfpu_binary_pow.h"       // calculate_sfpu_binary_pow / sfpu_binary_pow_init
#include "llk_sfpu/ckernel_sfpu_binary_remainder.h" // calculate_sfpu_binary_remainder / calculate_remainder_int32
#include "llk_sfpu/ckernel_sfpu_clamped_silu_glu.h" // calculate_clamped_silu_glu (silu(min(gate, 10)) * clamp(up, -10, 10))
#include "llk_sfpu/ckernel_sfpu_copy_dest_values.h" // copy_dest_value / copy_dest_value_init (Dest-to-Dest copy)
#include "llk_sfpu/ckernel_sfpu_div_int32.h"        // calculate_div_int32 / div_init (int32 / int32 -> fp32)
#include "llk_sfpu/ckernel_sfpu_div_int32_floor.h"  // calculate_div_int32_trunc / calculate_div_int32_floor
#include "llk_sfpu/ckernel_sfpu_int_sum.h"          // add_int (Dest tile += the next tile) / sum_int_init
#include "llk_sfpu/ckernel_sfpu_isclose.h"          // calculate_sfpu_isclose / isclose_init
#include "llk_sfpu/ckernel_sfpu_logaddexp.h"        // calculate_sfpu_logaddexp / calculate_sfpu_logaddexp_init
#include "llk_sfpu/ckernel_sfpu_logaddexp2.h"       // calculate_sfpu_logaddexp2 / calculate_sfpu_logaddexp2_init
#include "llk_sfpu/ckernel_sfpu_logsigmoid.h"       // calculate_logsigmoid (x, exp(-x) -> logsigmoid(x))
#include "llk_sfpu/ckernel_sfpu_mask.h"             // calculate_mask / calculate_mask_posinf / calculate_int_mask
#include "llk_sfpu/ckernel_sfpu_quant.h"            // quant_family / quant_family_init (quant/requant/dequant)
#include "llk_sfpu/ckernel_sfpu_shift.h"            // calculate_binary_left_shift / right / logical right
#include "llk_sfpu/ckernel_sfpu_situ_glu.h"         // calculate_situ_glu (softcapped gate * sigmoid(gate) * softcapped up)
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_binary_comp.h" // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"   // _mul_int32_ (int mul)

// Ternary SFPU op headers (consumed by the ternary dispatchers below).
// To add a new Quasar ternary SFPU op:
// 1. Include its ckernel header below.
// 2. Add the SfpuType enumerator if it is not there.
// 3. Add the `if constexpr` branch in call_ternary_sfpu_operation_quasar()
//    and init_ternary_sfpu_operation_quasar().
#include "llk_sfpu/ckernel_sfpu_where.h"
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"

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
inline constexpr bool is_zero_comp_op(SfpuType op)
{
    return op == SfpuType::equal_zero || op == SfpuType::not_equal_zero || op == SfpuType::less_than_zero || op == SfpuType::greater_than_zero ||
           op == SfpuType::less_than_equal_zero || op == SfpuType::greater_than_equal_zero;
}

/**
 * @brief Whether OPERATION is one of the trigonometry / inverse-hyperbolic ops.
 *
 * They share one init (@ref init_trigonometry, which programs ADDR_MOD_6 for the
 * auto-incrementing Dest store) since every trig body has the same load/compute/store shape.
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_trig_op(SfpuType op)
{
    return op == SfpuType::sine || op == SfpuType::cosine || op == SfpuType::acosh || op == SfpuType::asinh || op == SfpuType::atanh;
}

/**
 * @brief Whether OPERATION is one of the isinf / isnan predicates (one templated kernel).
 *
 * @param op The SFPU operation type to classify.
 */
inline constexpr bool is_isinf_isnan_op(SfpuType op)
{
    return op == SfpuType::isinf || op == SfpuType::isposinf || op == SfpuType::isneginf || op == SfpuType::isnan || op == SfpuType::isfinite;
}

/**
 * @brief Run the per-operation init step for a Quasar unary SFPU op.
 *
 * @tparam OPERATION The SFPU operation type (compile-time `SfpuType` constant).
 * @note Pair with @ref call_unary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuType OPERATION, bool is_fp32_dest_acc_en, bool APPROX = false>
void init_unary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuType::gelu)
    {
        gelu_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::abs || OPERATION == SfpuType::abs_int32)
    {
        abs_init();
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        init_square();
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        _init_rsqrt_<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        _init_reciprocal_<APPROX>();
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        init_zero_comp();
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        init_typecast();
    }
    else if constexpr (is_trig_op(OPERATION))
    {
        init_trigonometry<OPERATION, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::cumsum)
    {
        cumsum_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        hardsigmoid_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        celu_init();
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        elu_init();
    }
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        hardmish_init();
    }
    else if constexpr (OPERATION == SfpuType::mish)
    {
        mish_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::hardshrink)
    {
        hardshrink_init();
    }
    else if constexpr (OPERATION == SfpuType::hardtanh)
    {
        hardtanh_init();
    }
    else if constexpr (OPERATION == SfpuType::heaviside)
    {
        heaviside_init();
    }
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        prelu_init();
    }
    else if constexpr (OPERATION == SfpuType::selu)
    {
        selu_init();
    }
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        softshrink_init();
    }
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        init_softsign<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::sigmoid_appx)
    {
        sigmoid_appx_init();
    }
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        tanhshrink_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        xielu_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        cube_root_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        exp2_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        expm1_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        sfpu_binary_pow_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::sign)
    {
        sign_init();
    }
    else if constexpr (OPERATION == SfpuType::power)
    {
        sfpu_unary_pow_init();
    }
    else if constexpr (OPERATION == SfpuType::power_iterative)
    {
        power_init();
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        log_init<APPROX, false /* FAST_APPROX */, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        digamma_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::erf)
    {
        erf_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::erfc)
    {
        erfc_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        erfinv_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::i0)
    {
        i0_init();
    }
    else if constexpr (OPERATION == SfpuType::i1)
    {
        i1_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        lgamma_stirling_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        polygamma_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        logical_not_unary_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_gt)
    {
        unary_gt_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_lt)
    {
        unary_lt_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_ge)
    {
        unary_ge_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_le)
    {
        unary_le_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_eq)
    {
        unary_eq_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_ne)
    {
        unary_ne_init();
    }
    else if constexpr (OPERATION == SfpuType::bitwise_not)
    {
        bitwise_not_init();
    }
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        left_shift_init();
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        right_shift_init();
    }
    else if constexpr (OPERATION == SfpuType::bitwise_and)
    {
        bitwise_and_init();
    }
    else if constexpr (OPERATION == SfpuType::bitwise_or)
    {
        bitwise_or_init();
    }
    else if constexpr (OPERATION == SfpuType::bitwise_xor)
    {
        bitwise_xor_init();
    }
    // fmod / remainder read the divisor and its reciprocal from the programmable constants, set
    // here to 2.0 / 0.5 as in the Blackhole harness.
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        init_fmod<APPROX>(0x40000000u /* 2.0f */, 0x3f000000u /* 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        init_remainder<APPROX>(0x40000000u /* 2.0f */, 0x3f000000u /* 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::remainder_uint32)
    {
        remainder_uint32_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        rdiv_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::sum_int_col || OPERATION == SfpuType::sum_int_row)
    {
        sum_int_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::tiled_prod)
    {
        tiled_prod_init();
    }
    else if constexpr (OPERATION == SfpuType::alt_complex_rotate90)
    {
        alt_complex_rotate90_init();
    }
    else if constexpr (OPERATION == SfpuType::softcap)
    {
        softcap_init();
    }
    else if constexpr (OPERATION == SfpuType::tanh_derivative)
    {
        // tanh_derivative_tile's kernel: the accurate sech^2 form, whatever fast_and_approx says.
        tanh_derivative_sech2_init<APPROX>();
    }
    // rsub_scalar_int32 is stateless: its compute API init is SFPU_UNARY_INIT(unused).
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
template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, int ITERATIONS = SFPU_ITERATIONS>
void call_zero_comp_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format)
{
    static_assert(is_zero_comp_op(OPERATION), "call_zero_comp_operation_quasar: OPERATION must be a comparison-to-zero SfpuType");

    switch (sfpu_format)
    {
        case DataFormat::Int32:
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Int32, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::Int16:
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Int16, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::Int8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::Int8;
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, sfpu_fmt, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::UInt16:
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::UInt16, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        case DataFormat::UInt8:
        {
            constexpr DataFormat sfpu_fmt = is_fp32_dest_acc_en ? DataFormat::Int32 : DataFormat::UInt8;
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, sfpu_fmt, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
            break;
        }
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::Float32:
            // Float widths share the width-agnostic Float32 path: its sfpmem::DEFAULT access mode
            // resolves the actual width from ALU_FORMAT_SPEC_REG / ACC_CTRL.
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_zero_comp, (false, DataFormat::Float32, OPERATION, ITERATIONS), dst_index, VectorMode::RC);
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
 * @param first Whether this tile starts a fresh top-to-bottom accumulation chain; only cumsum
 *        reads it. Defaults to true so each tile is independent.
 * @note Must be preceded by @ref init_unary_sfpu_operation_quasar for the same op.
 */
template <
    SfpuType OPERATION,
    DstSync DST_SYNC,
    bool is_fp32_dest_acc_en,
    bool APPROX                    = false,
    int ITERATIONS                 = SFPU_ITERATIONS,
    DataFormat TYPECAST_IN_FORMAT  = DataFormat::Float32,
    DataFormat TYPECAST_OUT_FORMAT = DataFormat::Float16_b>
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32, [[maybe_unused]] const bool first = true)
{
    constexpr std::uint32_t kReluThresholdBits = 0x40A00000u; // 5.0f
    if constexpr (OPERATION == SfpuType::abs)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_abs, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::abs_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_abs_int32, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::fill)
    {
        // Fills with 5, the Blackhole harness's fill_const_value (and the golden's const_value): Int32
        // through the INT32 store, every float format through the float fill.
        if (sfpu_format == DataFormat::Int32)
        {
            SFPU_UNARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                _calculate_fill_int_,
                (APPROX, ckernel::InstrModLoadStore::INT32, ITERATIONS),
                dst_index,
                VectorMode::RC,
                5u /* fill value */);
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_fill_, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 5.0f /* fill value */);
        }
    }
    else if constexpr (OPERATION == SfpuType::exponential)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_exponential,
            (APPROX, is_fp32_dest_acc_en, false, ITERATIONS),
            dst_index,
            VectorMode::RC,
            p_sfpu::kCONST_1_FP16B);
    }
    else if constexpr (OPERATION == SfpuType::gelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_gelu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::relu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_relu_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::lrelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_lrelu_, (ITERATIONS), dst_index, VectorMode::RC, 0x3dcccccdu /* slope: 0.1f */);
    }
    else if constexpr (OPERATION == SfpuType::relu_min)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            _relu_min_,
            (sfpi::vFloat, APPROX, ITERATIONS, std::uint32_t),
            dst_index,
            VectorMode::RC,
            kReluThresholdBits /* threshold: 5.0f */);
    }
    else if constexpr (OPERATION == SfpuType::relu_max)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            _relu_max_,
            (sfpi::vFloat, APPROX, ITERATIONS, std::uint32_t),
            dst_index,
            VectorMode::RC,
            kReluThresholdBits /* threshold: 5.0f */);
    }
    else if constexpr (OPERATION == SfpuType::reciprocal)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_reciprocal, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::sqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sqrt_, (true /* APPROX */, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::tanh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sigmoid_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::silu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_silu_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::rsqrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rsqrt, (APPROX, ITERATIONS, is_fp32_dest_acc_en), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::square)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_square, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (is_trig_op(OPERATION))
    {
        // One op-templated kernel serves sine/cosine/acosh/asinh/atanh; OPERATION picks the branch
        // at compile time. APPROXIMATION_MODE=false selects the full-polynomial (accurate) path.
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_trigonometry, (OPERATION, false /* APPROX */, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::negative)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_negative_, (false, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::softplus)
    {
        // Softplus params beta / (1/beta) / threshold as fp32 bit patterns, matching the
        // UnarySFPUGolden._softplus reference defaults (beta = 1.0, threshold = 20.0).
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_softplus,
            (false, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            static_cast<std::uint32_t>(0x3F800000),  // beta = 1.0 (fp32)
            static_cast<std::uint32_t>(0x3F800000),  // 1/beta = 1.0 (fp32)
            static_cast<std::uint32_t>(0x41A00000)); // threshold = 20.0 (fp32)
    }
    else if constexpr (OPERATION == SfpuType::clamp)
    {
        // Clamp bounds fixed to [-1.0, +1.0] as fp32 bit patterns (matching the UnarySFPUGolden._clamp
        // reference). Extra args are forwarded to the per-face functor call.
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_clamp,
            (false, ITERATIONS),
            dst_index,
            VectorMode::RC,
            static_cast<std::uint32_t>(0xBF800000),  // min = -1.0 (fp32)
            static_cast<std::uint32_t>(0x3F800000)); // max = +1.0 (fp32)
    }
    else if constexpr (is_zero_comp_op(OPERATION))
    {
        call_zero_comp_operation_quasar<OPERATION, DST_SYNC, is_fp32_dest_acc_en, ITERATIONS>(dst_index, sfpu_format);
    }
    else if constexpr (OPERATION == SfpuType::typecast)
    {
        if constexpr (TYPECAST_IN_FORMAT == DataFormat::Float32 && TYPECAST_OUT_FORMAT == DataFormat::UInt16)
        {
            // Dedicated TTI kernel: names the FP32 load and UInt16 store formats explicitly
            // rather than letting HW imply them, so it needs implied math format disabled.
            // Walks Dest through ADDR_MOD_7 + _incr_counters_ instead of ADDR_MOD_6.
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_typecast_fp32_to_uint16_, (ITERATIONS), dst_index, VectorMode::RC);
        }
        else
        {
            // Same functor typecast_tile uses. Int32 → Float16_b is dispatched inside
            // calculate_typecast to _calculate_typecast_int32_to_fp16b_.
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_typecast, (TYPECAST_IN_FORMAT, TYPECAST_OUT_FORMAT, ITERATIONS), dst_index, VectorMode::RC);
        }
    }
    else if constexpr (OPERATION == SfpuType::cumsum)
    {
        // Whole-tile op: the accumulation chain spans all 32 tile rows and crosses the face-pair
        // boundary, so it runs once per tile (RC_custom), not once per face.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cumsum, (APPROX, ITERATIONS), dst_index, VectorMode::RC_custom, first);
    }
    else if constexpr (OPERATION == SfpuType::floor)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_floor_, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::ceil)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_ceil_, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::trunc)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_trunc_, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::frac)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_frac_, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::round)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_round_, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0 /* decimals */);
    }
    else if constexpr (OPERATION == SfpuType::add1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_add1, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    // Activation family. The fixed parameters match the Blackhole harness (sfpu_operations.h) and
    // helpers/sfpu_dispatch_constants.py, which the goldens read.
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_activation, (APPROX, ckernel::ActivationType::Hardsigmoid, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_celu,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x3f800000u /* alpha = 1.0f */,
            0x3f800000u /* 1/alpha = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_elu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x3f800000u /* alpha = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, hardmish, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::mish)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_mish, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::hardshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_hardshrink, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* lambda = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::hardtanh)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_hardtanh,
            (APPROX, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0xBF800000u /* min = -1.0f */,
            0x3F800000u /* max = 1.0f */);
    }
    else if constexpr (OPERATION == SfpuType::heaviside)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_heaviside, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_prelu, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3e800000u /* slope = 0.25f */);
    }
    else if constexpr (OPERATION == SfpuType::selu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_selu,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x3f867d5fu /* scale ~= 1.0507 */,
            0x3fd62d7du /* alpha ~= 1.6733 */);
    }
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softshrink, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* lambda = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softsign, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::sigmoid_appx)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sigmoid_appx, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanhshrink, (is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_xielu,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x3f800000u /* alpha_p = 1.0f */,
            0x3f800000u /* alpha_n = 1.0f */);
    }
    // Elementary-math family.
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cube_root, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_exp2, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_expm1, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_rpow, (APPROX, ITERATIONS, is_fp32_dest_acc_en), dst_index, VectorMode::RC, 0x40000000u /* base = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuType::sign)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sign, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0u /* exponent_size_8 */);
    }
    else if constexpr (OPERATION == SfpuType::power)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_unary_power,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x40000000u /* exponent = 2.0f */);
    }
    else if constexpr (OPERATION == SfpuType::power_iterative)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_unary_power_iterative,
            (APPROX, ITERATIONS),
            dst_index,
            VectorMode::RC,
            3u /* exponent */);
    }
    else if constexpr (OPERATION == SfpuType::log)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_log,
            (APPROX, false /* FAST_APPROX */, false /* HAS_BASE_SCALING */, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0u /* log_base_scale_factor (unused: HAS_BASE_SCALING = false) */);
    }
    // Special-function family.
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_digamma, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::erf)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erf, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::erfc)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfc, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::erfinv)
    {
        // calculate_erfinv fixes its own per-face ITERATIONS = 8, as on Blackhole.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfinv, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::i0)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i0, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::i1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i1, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_lgamma_stirling, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        // order n = 1 (trigamma); scale = (-1)^(n+1) * n! = 1.0f.
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_polygamma,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x3f800000u /* n = 1.0f */,
            0x3f800000u /* scale = 1.0f */);
    }
    // Comparison / logical family.
    else if constexpr (is_isinf_isnan_op(OPERATION))
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (OPERATION, APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        // logical_not(x) = (x == 0) ? 1 : 0; the layout follows the Dest format, as on Blackhole.
        if (sfpu_format == DataFormat::Int32)
        {
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, ckernel::InstrModLoadStore::INT32, ITERATIONS), dst_index, VectorMode::RC);
        }
        else if (sfpu_format == DataFormat::UInt16)
        {
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, ckernel::InstrModLoadStore::LO16, ITERATIONS), dst_index, VectorMode::RC);
        }
        else
        {
            SFPU_UNARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, ckernel::InstrModLoadStore::DEFAULT, ITERATIONS), dst_index, VectorMode::RC);
        }
    }
    else if constexpr (OPERATION == SfpuType::unary_gt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_gt, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_lt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_lt, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_ge)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_ge, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_le)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_le, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_eq)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_eq, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::unary_ne)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_ne, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3f000000u /* value = 0.5f */);
    }
    else if constexpr (OPERATION == SfpuType::threshold)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            _calculate_threshold_,
            (APPROX, ITERATIONS, float),
            dst_index,
            VectorMode::RC,
            5.0f /* threshold */,
            10.0f /* replacement value */);
    }
    else if constexpr (OPERATION == SfpuType::bitwise_not)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_bitwise_not, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    // Integer-only kernels, driven as Int32 like the Blackhole harness: shift by 3 bits, bitwise
    // and/or/xor with 0x70FF00F5 and rsub from INT_MAX (sfpu_dispatch_constants.py holds the golden's
    // copy). Both scalars are non-negative: the harness stores Int32 in L1 as sign-magnitude, so only
    // non-negative results can be compared with these two's-complement kernels.
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_left_shift, (APPROX, DataFormat::Int32, ITERATIONS), dst_index, VectorMode::RC, 3u /* shift */);
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_right_shift, (APPROX, DataFormat::Int32, ITERATIONS), dst_index, VectorMode::RC, 3u /* shift */);
    }
    else if constexpr (OPERATION == SfpuType::bitwise_and || OPERATION == SfpuType::bitwise_or || OPERATION == SfpuType::bitwise_xor)
    {
        constexpr UnaryBitwiseOp BW = (OPERATION == SfpuType::bitwise_and)  ? UnaryBitwiseOp::AND
                                      : (OPERATION == SfpuType::bitwise_or) ? UnaryBitwiseOp::OR
                                                                            : UnaryBitwiseOp::XOR;
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (APPROX, BW, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x70FF00F5u /* scalar */);
    }
    else if constexpr (OPERATION == SfpuType::rsub_scalar_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rsub_scalar_int32, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x7FFFFFFFu /* scalar */);
    }
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        // The divisor comes from init_fmod(), so no runtime argument.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_fmod, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        // The divisor comes from init_remainder(), so no runtime argument.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_remainder, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::remainder_uint32)
    {
        // Unsigned x mod 1000 on the 32-bit pattern (Int32 in Dest: Quasar has no UInt32); 1000 takes
        // the general range-reduce branch (sfpu_dispatch_constants.py holds the golden's copy).
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_remainder_uint32_scalar, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 1000u /* divisor */);
    }
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_rdiv,
            (APPROX, is_fp32_dest_acc_en, ckernel::RoundingMode::None, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x40000000u /* value = 2.0f */);
    }
    // int_sum's partial reductions read fixed offsets into the faces below / beside, so they run
    // on faces 0, 1 (col, VectorMode::R) and 0, 2 (row, VectorMode::C) only, as the compute API calls them.
    else if constexpr (OPERATION == SfpuType::sum_int_col)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sum_int_col, (APPROX), dst_index, VectorMode::R);
    }
    else if constexpr (OPERATION == SfpuType::sum_int_row)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sum_int_row, (APPROX), dst_index, VectorMode::C);
    }
    else if constexpr (OPERATION == SfpuType::tiled_prod)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tiled_prod, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::identity)
    {
        // Integer Dest takes the bit-exact vUInt copy, as the compute API's identity_tile_uint32 does.
        if (sfpu_format == DataFormat::Int32)
        {
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_identity_uint, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
        }
        else
        {
            SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_identity, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
        }
    }
    else if constexpr (OPERATION == SfpuType::cast_fp32_to_fp16a)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, cast_fp32_to_fp16a, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::alt_complex_rotate90)
    {
        // Each iteration covers both column halves of a 4-row group (dst_reg += 2), so a face takes
        // the kernel's default 4 iterations, as in the compute API; ITERATIONS would run past the face.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::softcap)
    {
        // beta = 5 and its reciprocal, as fp32 bits (sfpu_dispatch_constants.py holds the golden's copy).
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_softcap,
            (APPROX, is_fp32_dest_acc_en, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x40A00000u /* beta = 5.0f */,
            0x3E4CCCCDu /* 1 / beta = 0.2f */);
    }
    else if constexpr (OPERATION == SfpuType::tanh_derivative)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanh_derivative_sech2, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
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
    else if constexpr (OP == BinaryOp::ISCLOSE)
    {
        isclose_init();
    }
    else if constexpr (OP == BinaryOp::FMOD)
    {
        fmod_binary_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::REMAINDER)
    {
        remainder_binary_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::POW)
    {
        sfpu_binary_pow_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::FMOD_INT32)
    {
        fmod_int32_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::REMAINDER_INT32)
    {
        remainder_int32_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::REMAINDER_UINT32)
    {
        remainder_uint32_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::LGAMMA_STIRLING_FP32)
    {
        lgamma_stirling_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::MASK || OP == BinaryOp::MASK_POSINF || OP == BinaryOp::INT_MASK)
    {
        mask_init();
    }
    else if constexpr (OP == BinaryOp::DIV_INT32)
    {
        div_trunc_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::DIV_INT32_FLOOR)
    {
        div_floor_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::INT_SUM_ADD)
    {
        sum_int_init<APPROXIMATION_MODE>();
    }
    else if constexpr (OP == BinaryOp::CLAMPED_SILU_GLU)
    {
        clamped_silu_glu_init();
    }
    else if constexpr (OP == BinaryOp::SITU_GLU)
    {
        situ_glu_init();
    }
    else if constexpr (OP == BinaryOp::ADD_TOP_ROW)
    {
        init_add_top_row();
    }
    else if constexpr (OP == BinaryOp::LOGADDEXP)
    {
        // log1p's coefficients live in the program constant registers and differ by destination
        // precision; the init also programs the ADDR_MOD_6 the store advances through.
        calculate_sfpu_logaddexp_init<is_fp32_dest_acc_en>();
    }
    else if constexpr (OP == BinaryOp::LOGADDEXP2)
    {
        calculate_sfpu_logaddexp2_init<is_fp32_dest_acc_en>();
    }
    // RSHFT / LSHFT / LOGICAL_RSHFT need no init beyond the shared SFPU one.
    // ADD / SUB / GT / LT / LE / GE / COPY_DEST / LOGSIGMOID are stateless — no init.
}

/**
 * @brief Apply a Quasar binary SFPU op over Dest operands into a result tile.
 *
 * Most ops read two Dest operands (`src0_tile`, `src1_tile`) into `dst_tile`.
 * COPY_DEST is a Dest-to-Dest copy of `src0_tile` onto `dst_tile`; `src1_tile` is
 * ignored and the kernel ABI is `(in, out, unused)` to match the shared compute API.
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
 * @param src0_tile,src1_tile,dst_tile Operand / result tile indices. COPY_DEST ignores
 *        `src1_tile` and writes `src0_tile` onto `dst_tile`.
 * @param math_format Dest encoding. Int32 vs float path for MUL and max/min; COPY_DEST
 *        forwards it as the `copy_dest_value` template argument so the sfpmem mode
 *        matches Dest (Float16 / Float16_b / Int32 / UInt16 / …, not a Float32 placeholder).
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
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_add_int,
                (false, ITERATIONS, DataFormat::Int32, 0, false),
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
                calculate_sfpu_binary,
                (APPROXIMATION_MODE, BinaryOp::ADD, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
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
            (APPROXIMATION_MODE, BinaryOp::SUB, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
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
                (APPROXIMATION_MODE, BinaryOp::MUL, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
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
            (APPROXIMATION_MODE, BinaryOp::DIV, is_fp32_dest_acc_en, dst_rounding_mode, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        // atan2(y, x): src0 = y, src1 = x. is_fp32_dest_acc_en must match the init's.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_atan2,
            (APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::COPY_DEST)
    {
        // Dest-to-Dest copy of src0 onto dst. Kernel ABI is (in, out, unused), matching
        // the shared compute API — not the usual binary (in0, in1, out). src1_tile is
        // ignored. math_format is the Dest encoding and must be forwarded so
        // copy_dest_value can pick the matching sfpmem mode via _sfpu_sfpmem_type_
        // (Int32 → INT32 for TEN-4674; UInt16/Int16/Int8/UInt8 keep dedicated modes).
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::Int32, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
        else if (math_format == DataFormat::Float16)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::Float16, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
        else if (math_format == DataFormat::Float16_b)
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                copy_dest_value,
                (DataFormat::Float16_b, false, ITERATIONS),
                src0_tile,
                dst_tile,
                0 /* unused */,
                VectorMode::RC);
        }
        else if (math_format == DataFormat::UInt16)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::UInt16, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
        else if (math_format == DataFormat::Int16)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::Int16, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
        else if (math_format == DataFormat::Int8)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::Int8, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
        else if (math_format == DataFormat::UInt8)
        {
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::UInt8, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
        else
        {
            // Float32 and Tf32 both map to sfpmem::FP32 inside copy_dest_value.
            SFPU_BINARY_CALL(
                DST_SYNC, is_fp32_dest_acc_en, copy_dest_value, (DataFormat::Float32, false, ITERATIONS), src0_tile, dst_tile, 0 /* unused */, VectorMode::RC);
        }
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
    else if constexpr (OP == BinaryOp::FMOD)
    {
        // Float fmod (result sign follows the dividend); is_fp32_dest_acc_en selects the store.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_fmod,
            (APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::REMAINDER)
    {
        // Float remainder (result sign follows the divisor).
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_remainder,
            (APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::POW)
    {
        // pow(base = src0, exponent = src1), the binary_pow kernel's own entry point.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_pow,
            (APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::BITWISE_AND || OP == BinaryOp::BITWISE_OR || OP == BinaryOp::BITWISE_XOR)
    {
        // int32 bitwise AND/OR/XOR on the raw 32-bit patterns (INT32 layout).
        constexpr BinaryBitwiseOp BW = (OP == BinaryOp::BITWISE_AND)  ? BinaryBitwiseOp::AND
                                       : (OP == BinaryOp::BITWISE_OR) ? BinaryBitwiseOp::OR
                                                                      : BinaryBitwiseOp::XOR;
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_bitwise,
            (APPROXIMATION_MODE, BW, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::FMOD_INT32)
    {
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_fmod_int32, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::REMAINDER_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_remainder_int32, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::REMAINDER_UINT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_remainder_uint32, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LGAMMA_STIRLING_FP32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_lgamma_stirling_fp32, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::RSUB_INT32)
    {
        // out = in1 - in0, two's complement, INT32 layout as in the Blackhole harness.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_rsub_int,
            (APPROXIMATION_MODE, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::DIV_INT32)
    {
        // int32 truncating division (rounds toward zero), as in the Blackhole harness.
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_div_int32_trunc, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::DIV_INT32_FLOOR)
    {
        // int32 floor division (rounds toward -inf).
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_div_int32_floor, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::INT_SUM_ADD)
    {
        // add_int reads the tile it is pointed at and the one after it (dst_reg[32]) and writes
        // in place, whatever its dst_offset argument says, so it fits the binary harness only
        // with in1 = in0 + 1 and out = in0 (the compute API's sfpu_add_int(dst0, dst1) contract).
        LLK_ASSERT(src1_tile == src0_tile + 1 && dst_tile == src0_tile, "INT_SUM_ADD needs in1 = in0 + 1 and out = in0");
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, add_int, (APPROXIMATION_MODE, ITERATIONS), src0_tile, VectorMode::RC, src1_tile - src0_tile /* dst_offset */);
    }
    else if constexpr (OP == BinaryOp::CLAMPED_SILU_GLU)
    {
        // gate = src0, up = src1; the clamp limit is the kernel's compile-time DeepSeek-V4 config.
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_clamped_silu_glu, (is_fp32_dest_acc_en, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::SITU_GLU)
    {
        // gate = src0, up = src1; the betas are the kernel's compile-time Kimi config.
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_situ_glu, (is_fp32_dest_acc_en, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::ADD_TOP_ROW)
    {
        // One call per tile (VectorMode::None, Blackhole's RC_custom): the kernel reaches the top
        // four rows of faces 0 and 1 itself. The format is a template argument, so dispatch on it.
        if (math_format == DataFormat::Int32)
        {
            SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_add_top_row, (DataFormat::Int32), src0_tile, src1_tile, dst_tile, VectorMode::None);
        }
        else
        {
            LLK_ASSERT(math_format == DataFormat::Float32, "ADD_TOP_ROW supports Float32 and Int32");
            SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_add_top_row, (DataFormat::Float32), src0_tile, src1_tile, dst_tile, VectorMode::None);
        }
    }
    else if constexpr (OP == BinaryOp::LOGADDEXP)
    {
        // is_fp32_dest_acc_en selects the fp32 or bf16 exponential, the log1p coefficient set the
        // paired init loaded, and the bf16 rounding before the store.
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_logaddexp,
            (APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LOGADDEXP2)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_logaddexp2,
            (APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::RSHFT || OP == BinaryOp::LSHFT || OP == BinaryOp::LOGICAL_RSHFT)
    {
        // INT32, not INT32_2S_COMP, as in the Blackhole harness and binary_shift.h: Int32 tiles reach
        // Dest raw, so the shift operates on the bits directly. a = src0, shift amount = src1.
        if constexpr (OP == BinaryOp::RSHFT)
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_binary_right_shift,
                (APPROXIMATION_MODE, ITERATIONS, ckernel::InstrModLoadStore::INT32, false),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
        else if constexpr (OP == BinaryOp::LSHFT)
        {
            SFPU_BINARY_CALL(
                DST_SYNC,
                is_fp32_dest_acc_en,
                calculate_binary_left_shift,
                (APPROXIMATION_MODE, ITERATIONS, ckernel::InstrModLoadStore::INT32, false),
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
                calculate_logical_right_shift,
                (APPROXIMATION_MODE, ITERATIONS, ckernel::InstrModLoadStore::INT32, false),
                src0_tile,
                src1_tile,
                dst_tile,
                VectorMode::RC);
        }
    }
    else if constexpr (OP == BinaryOp::ISCLOSE)
    {
        // isclose(a, b) = |a - b| <= atol + rtol * |b| with torch's defaults rtol = 1e-5, atol = 1e-8
        // and equal_nan = false, as in the Blackhole harness (and the golden).
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_isclose,
            (APPROXIMATION_MODE, ITERATIONS, /*EQUAL_NAN=*/false),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC,
            0x3727c5acu /* rtol = 1e-5f */,
            0x322bcc77u /* atol = 1e-8f */);
    }
    else if constexpr (OP == BinaryOp::MASK)
    {
        // The mask kernels read data from the tile they are pointed at and the mask from the one
        // after it (dst_reg[32]), and write in place, so position Dest at src0_tile.
        LLK_ASSERT(dst_tile == src0_tile && src1_tile == src0_tile + 1, "Mask requires adjacent inputs and an in-place output");
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_mask, (APPROXIMATION_MODE, ITERATIONS), src0_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::MASK_POSINF)
    {
        // The mask kernels read data from the tile they are pointed at and the mask from the one
        // after it (dst_reg[32]), and write in place, so position Dest at src0_tile.
        LLK_ASSERT(dst_tile == src0_tile && src1_tile == src0_tile + 1, "Mask requires adjacent inputs and an in-place output");
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_mask_posinf, (APPROXIMATION_MODE, ITERATIONS), src0_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::INT_MASK)
    {
        // The mask kernels read data from the tile they are pointed at and the mask from the one
        // after it (dst_reg[32]), and write in place, so position Dest at src0_tile.
        LLK_ASSERT(dst_tile == src0_tile && src1_tile == src0_tile + 1, "Mask requires adjacent inputs and an in-place output");
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_int_mask, (APPROXIMATION_MODE, ITERATIONS), src0_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::LOGSIGMOID)
    {
        // logsigmoid(x) = -softplus(-x): src0 holds x, src1 holds exp(-x), which the caller computes
        // first (the YAML case runs Neg + Exp on a copy of x), as the Blackhole compute path does.
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_logsigmoid, (APPROXIMATION_MODE, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
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

/**
 * @brief Initialize shared SFPU state and the selected Quasar ternary operation.
 *
 * @tparam OPERATION Ternary SFPU operation to initialize.
 * @tparam is_fp32_dest_acc_en Dest accumulation mode, matching the calculate step.
 * @tparam APPROX Approximation mode, matching the calculate step.
 * @note Pair with @ref call_ternary_sfpu_operation_quasar for the calculate step.
 */
template <SfpuType OPERATION, bool is_fp32_dest_acc_en, bool APPROX = false>
void init_ternary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuType::where)
    {
        _llk_math_eltwise_ternary_sfpu_init_<OPERATION>();
    }
    else
    {
        static_assert(unhandled_op<OPERATION>, "init_ternary_sfpu_operation_quasar: unhandled Quasar ternary SFPU operation");
    }
}

/**
 * @brief Apply a Quasar ternary SFPU op over three Dest operands into a result tile.
 *
 * @tparam OPERATION Ternary SFPU operation to execute.
 * @tparam DST_SYNC Destination synchronization mode used for bounds checking.
 * @tparam is_fp32_dest_acc_en Whether Dest is in FP32 mode.
 * @tparam APPROX Whether to use the operation's approximate path.
 * @tparam ITERATIONS Number of SFPU row-pair iterations per face.
 * @param src0_tile First operand tile index; the condition for where.
 * @param src1_tile Second operand tile index; the true value for where.
 * @param src2_tile Third operand tile index; the false value for where.
 * @param dst_tile Result tile index, which may alias an input.
 * @param vector_mode Faces to process; defaults to the whole tile.
 * @note Call @ref init_ternary_sfpu_operation_quasar for the same op before this function.
 */
template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, bool APPROX = false, int ITERATIONS = SFPU_ITERATIONS>
void call_ternary_sfpu_operation_quasar(
    const std::uint32_t src0_tile,
    const std::uint32_t src1_tile,
    const std::uint32_t src2_tile,
    const std::uint32_t dst_tile,
    VectorMode vector_mode = VectorMode::RC)
{
    if constexpr (OPERATION == SfpuType::where)
    {
        SFPU_TERNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_where, (APPROX, ITERATIONS), src0_tile, src1_tile, src2_tile, dst_tile, vector_mode);
    }
    else
    {
        static_assert(unhandled_op<OPERATION>, "call_ternary_sfpu_operation_quasar: unhandled Quasar ternary SFPU operation");
    }
}

} // namespace test_utils
