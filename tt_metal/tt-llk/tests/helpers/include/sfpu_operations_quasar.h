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
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_activations.h"
#include "llk_sfpu/ckernel_sfpu_add1.h"
#include "llk_sfpu/ckernel_sfpu_addcdiv.h"
#include "llk_sfpu/ckernel_sfpu_addcmul.h"
#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_atan2.h"
#include "llk_sfpu/ckernel_sfpu_binary_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_binary_fmod.h"
#include "llk_sfpu/ckernel_sfpu_binary_pow.h"
#include "llk_sfpu/ckernel_sfpu_binary_remainder.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_bitwise_not.h"
#include "llk_sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h"
#include "llk_sfpu/ckernel_sfpu_cbrt.h"
#include "llk_sfpu/ckernel_sfpu_celu.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_digamma.h"
#include "llk_sfpu/ckernel_sfpu_div_int32_floor.h"
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
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/ckernel_sfpu_isclose.h"
#include "llk_sfpu/ckernel_sfpu_lerp.h"
#include "llk_sfpu/ckernel_sfpu_lgamma.h"
#include "llk_sfpu/ckernel_sfpu_logical_not.h"
#include "llk_sfpu/ckernel_sfpu_logsigmoid.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
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
#include "llk_sfpu/ckernel_sfpu_sign.h"
#include "llk_sfpu/ckernel_sfpu_snake_beta.h"
#include "llk_sfpu/ckernel_sfpu_softplus.h"
#include "llk_sfpu/ckernel_sfpu_softshrink.h"
#include "llk_sfpu/ckernel_sfpu_softsign.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/ckernel_sfpu_tanh.h"
#include "llk_sfpu/ckernel_sfpu_tanhshrink.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"
#include "llk_sfpu/ckernel_sfpu_trigonometry.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "llk_sfpu/ckernel_sfpu_unary_comp.h"
#include "llk_sfpu/ckernel_sfpu_unary_power.h"
#include "llk_sfpu/ckernel_sfpu_unary_shift.h"
#include "llk_sfpu/ckernel_sfpu_where.h"
#include "llk_sfpu/ckernel_sfpu_xielu.h"
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
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_add.h"         // _add_int_ (int add)
#include "sfpu/ckernel_sfpu_binary_comp.h" // calculate_binary_comp_int32 (int gt/lt/le/ge)
#include "sfpu/ckernel_sfpu_mul_int32.h"   // _mul_int32_ (int mul)

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

inline constexpr bool is_multi_tile_unary_op(SfpuType op)
{
    return op == SfpuType::mask || op == SfpuType::int_mask;
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
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        hardsigmoid_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        cube_root_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        celu_init();
    }
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        digamma_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        elu_init();
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
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        exp2_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        expm1_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        init_fmod<APPROX>(0x40000000u, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        hardmish_init();
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
        lgamma_stirling_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        logical_not_unary_init();
    }
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        polygamma_init<APPROX, is_fp32_dest_acc_en>();
    }
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        prelu_init();
    }
    else if constexpr (OPERATION == SfpuType::rdiv)
    {
        rdiv_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        init_remainder<APPROX>(0x40000000u, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        sfpu_binary_pow_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::selu)
    {
        selu_init();
    }
    else if constexpr (OPERATION == SfpuType::sign)
    {
        sign_init();
    }
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        softshrink_init();
    }
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        init_softsign<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        tanhshrink_init<APPROX, is_fp32_dest_acc_en>();
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
    else if constexpr (OPERATION == SfpuType::power)
    {
        sfpu_unary_pow_init();
    }
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        left_shift_init();
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        right_shift_init();
    }
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        xielu_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::alt_complex_rotate90)
    {
        alt_complex_rotate90_init();
    }
    else if constexpr (OPERATION == SfpuType::int_sum_col || OPERATION == SfpuType::int_sum_row)
    {
        sum_int_init<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::unary_bitwise_and)
    {
        bitwise_and_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_bitwise_or)
    {
        bitwise_or_init();
    }
    else if constexpr (OPERATION == SfpuType::unary_bitwise_xor)
    {
        bitwise_xor_init();
    }
    else if constexpr (OPERATION == SfpuType::mask || OPERATION == SfpuType::int_mask)
    {
        mask_init();
    }
    else if constexpr (OPERATION == SfpuType::tiled_prod)
    {
        tiled_prod_init();
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
void call_unary_sfpu_operation_quasar(std::uint32_t dst_index, DataFormat sfpu_format = DataFormat::Float32)
{
    if constexpr (OPERATION == SfpuType::alt_complex_rotate90)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::int_sum_col)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sum_int_col, (APPROX), dst_index, VectorMode::R);
    }
    else if constexpr (OPERATION == SfpuType::int_sum_row)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sum_int_row, (APPROX), dst_index, VectorMode::C);
    }
    else if constexpr (OPERATION == SfpuType::unary_bitwise_and)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (APPROX, UnaryBitwiseOp::AND, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x55u);
    }
    else if constexpr (OPERATION == SfpuType::unary_bitwise_or)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (APPROX, UnaryBitwiseOp::OR, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x55u);
    }
    else if constexpr (OPERATION == SfpuType::unary_bitwise_xor)
    {
        SFPU_UNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (APPROX, UnaryBitwiseOp::XOR, DataFormat::Int32, ITERATIONS),
            dst_index,
            VectorMode::RC,
            0x55u);
    }
    else if constexpr (OPERATION == SfpuType::mask)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_mask, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::int_mask)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_int_mask, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::tiled_prod)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tiled_prod, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::abs)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _calculate_abs_, (ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::hardsigmoid)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_activation, (APPROX, ckernel::ActivationType::Hardsigmoid, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::add1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_add1, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::bitwise_not)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_bitwise_not, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::cast_fp32_to_fp16a)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, cast_fp32_to_fp16a, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::cbrt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_cube_root, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::celu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_celu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x3F800000u, 0x3F800000u);
    }
    else if constexpr (OPERATION == SfpuType::digamma)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_digamma, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::elu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_elu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x3F800000u);
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
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_erfinv, (APPROX), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::exp2)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_exp2, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::expm1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_expm1, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::fmod)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_fmod, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::hardmish)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, hardmish, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::hardshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_hardshrink, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::hardtanh)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_hardtanh, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0xBF800000u, 0x3F800000u);
    }
    else if constexpr (OPERATION == SfpuType::heaviside)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_heaviside, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::i0)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i0, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::i1)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_i1, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::identity)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_identity, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::lgamma)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_lgamma_stirling, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::logical_not_unary)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, ckernel::InstrModLoadStore::DEFAULT, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::polygamma)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_polygamma, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x3F800000u, 0x3F800000u);
    }
    else if constexpr (OPERATION == SfpuType::prelu)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_prelu, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3E800000u);
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
            0x40000000u);
    }
    else if constexpr (OPERATION == SfpuType::remainder)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_remainder, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::rpow)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_rpow, (APPROX, ITERATIONS, is_fp32_dest_acc_en), dst_index, VectorMode::RC, 0x40000000u);
    }
    else if constexpr (OPERATION == SfpuType::selu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_selu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x3F867D5Fu, 0x3FD62D7Du);
    }
    else if constexpr (OPERATION == SfpuType::sign)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sign, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0u);
    }
    else if constexpr (OPERATION == SfpuType::softshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softshrink, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::softsign)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_softsign, (APPROX, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::tanhshrink)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_tanhshrink, (is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (OPERATION == SfpuType::unary_gt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_gt, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_lt)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_lt, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_ge)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_ge, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_le)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_le, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_eq)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_eq, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::unary_ne)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_unary_ne, (APPROX, ITERATIONS), dst_index, VectorMode::RC, 0x3F000000u);
    }
    else if constexpr (OPERATION == SfpuType::power)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_unary_power, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x40000000u);
    }
    else if constexpr (OPERATION == SfpuType::left_shift)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_left_shift, (APPROX, DataFormat::Int32, ITERATIONS), dst_index, VectorMode::RC, 3u);
    }
    else if constexpr (OPERATION == SfpuType::right_shift)
    {
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_right_shift, (APPROX, DataFormat::Int32, ITERATIONS), dst_index, VectorMode::RC, 3u);
    }
    else if constexpr (OPERATION == SfpuType::xielu)
    {
        SFPU_UNARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_xielu, (APPROX, is_fp32_dest_acc_en, ITERATIONS), dst_index, VectorMode::RC, 0x3F800000u, 0x3F800000u);
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
        // at compile time. APPROXIMATION_MODE=false selects the full-polynomial (accurate) path;
        // VectorMode::RC (the params default) runs the functor once per face.
        _llk_math_eltwise_unary_sfpu_params_(calculate_trigonometry<OPERATION, false /* APPROX */, is_fp32_dest_acc_en, ITERATIONS>, dst_index);
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
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_typecast, (TYPECAST_IN_FORMAT, TYPECAST_OUT_FORMAT, ITERATIONS), dst_index, VectorMode::RC);
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
template <ckernel::BinaryOp OP, bool SIGN_MAGNITUDE_FORMAT = false, bool is_fp32_dest_acc_en = true>
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
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        calculate_sfpu_atan2_init<false, is_fp32_dest_acc_en>();
    }
    else if constexpr (OP == BinaryOp::FMOD)
    {
        fmod_binary_init<false>();
    }
    else if constexpr (OP == BinaryOp::POW)
    {
        sfpu_binary_pow_init<false>();
    }
    else if constexpr (OP == BinaryOp::REMAINDER)
    {
        remainder_binary_init<false, is_fp32_dest_acc_en>();
    }
    else if constexpr (OP == BinaryOp::DIV_INT32)
    {
        div_trunc_init<false>();
    }
    else if constexpr (OP == BinaryOp::DIV_INT32_FLOOR)
    {
        div_floor_init<false>();
    }
    else if constexpr (OP == BinaryOp::ISCLOSE)
    {
        isclose_init();
    }
    else if constexpr (OP == BinaryOp::LOGSIGMOID)
    {
        logsigmoid_init<false>();
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
    else if constexpr (OP == BinaryOp::ATAN2)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_sfpu_atan2, (false, ITERATIONS, is_fp32_dest_acc_en), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::BITWISE_AND)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_bitwise,
            (false, BinaryBitwiseOp::AND, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::BITWISE_OR)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_bitwise,
            (false, BinaryBitwiseOp::OR, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::BITWISE_XOR)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_bitwise,
            (false, BinaryBitwiseOp::XOR, ckernel::InstrModLoadStore::INT32, ITERATIONS),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::FMOD)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_fmod,
            (false, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::POW)
    {
        SFPU_BINARY_CALL(
            DST_SYNC, is_fp32_dest_acc_en, calculate_sfpu_binary_pow, (false, ITERATIONS, is_fp32_dest_acc_en), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::REMAINDER)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_binary_remainder,
            (false, ITERATIONS, is_fp32_dest_acc_en),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::DIV_INT32)
    {
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_div_int32_trunc, (false, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::DIV_INT32_FLOOR)
    {
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_div_int32_floor, (false, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::ISCLOSE)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_sfpu_isclose,
            (false, ITERATIONS, false),
            src0_tile,
            src1_tile,
            dst_tile,
            VectorMode::RC,
            0x3727C5ACu,
            0x322BCC77u);
    }
    else if constexpr (OP == BinaryOp::LOGSIGMOID)
    {
        SFPU_BINARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_logsigmoid, (false, ITERATIONS), src0_tile, src1_tile, dst_tile, VectorMode::RC);
    }
    else if constexpr (OP == BinaryOp::RSUB_INT32)
    {
        SFPU_BINARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_rsub_int,
            (false, ckernel::InstrModLoadStore::INT32, ITERATIONS),
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

template <SfpuType OPERATION, bool is_fp32_dest_acc_en, bool APPROX = false>
void init_ternary_sfpu_operation_quasar()
{
    if constexpr (OPERATION == SfpuType::addcdiv)
    {
        init_addcdiv<APPROX>();
    }
    else if constexpr (OPERATION == SfpuType::snake_beta)
    {
        snake_beta_init<APPROX>();
    }
}

template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, DataFormat DATA_FORMAT, bool APPROX = false, int ITERATIONS = SFPU_ITERATIONS>
void call_ternary_sfpu_operation_quasar_impl(
    std::uint32_t in0_tile, std::uint32_t in1_tile, std::uint32_t in2_tile, std::uint32_t out_tile, VectorMode vector_mode, std::uint32_t scalar)
{
    if constexpr (OPERATION == SfpuType::where)
    {
        SFPU_TERNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_where, (APPROX), in0_tile, in1_tile, in2_tile, out_tile, vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::addcdiv)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_addcdiv,
            (APPROX, is_fp32_dest_acc_en, DATA_FORMAT, ITERATIONS),
            in0_tile,
            in1_tile,
            in2_tile,
            out_tile,
            vector_mode,
            scalar);
    }
    else if constexpr (OPERATION == SfpuType::addcmul)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_addcmul,
            (APPROX, is_fp32_dest_acc_en, DATA_FORMAT, ITERATIONS),
            in0_tile,
            in1_tile,
            in2_tile,
            out_tile,
            vector_mode,
            scalar);
    }
    else if constexpr (OPERATION == SfpuType::lerp)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_lerp,
            (APPROX, is_fp32_dest_acc_en, DATA_FORMAT, ITERATIONS),
            in0_tile,
            in1_tile,
            in2_tile,
            out_tile,
            vector_mode);
    }
    else if constexpr (OPERATION == SfpuType::snake_beta)
    {
        SFPU_TERNARY_CALL(
            DST_SYNC,
            is_fp32_dest_acc_en,
            calculate_snake_beta,
            (APPROX, is_fp32_dest_acc_en, DATA_FORMAT, ITERATIONS),
            in0_tile,
            in1_tile,
            in2_tile,
            out_tile,
            vector_mode);
    }
    else
    {
        static_assert(unhandled_op<OPERATION>, "Unhandled Quasar ternary SFPU operation");
    }
}

template <SfpuType OPERATION, DstSync DST_SYNC, bool is_fp32_dest_acc_en, bool APPROX = false, int ITERATIONS = SFPU_ITERATIONS>
void call_ternary_sfpu_operation_quasar(
    std::uint32_t in0_tile,
    std::uint32_t in1_tile,
    std::uint32_t in2_tile,
    std::uint32_t out_tile,
    VectorMode vector_mode,
    DataFormat data_format,
    std::uint32_t scalar)
{
    if (data_format == DataFormat::Float32)
    {
        call_ternary_sfpu_operation_quasar_impl<OPERATION, DST_SYNC, is_fp32_dest_acc_en, DataFormat::Float32, APPROX, ITERATIONS>(
            in0_tile, in1_tile, in2_tile, out_tile, vector_mode, scalar);
    }
    else if (data_format == DataFormat::Float16_b)
    {
        call_ternary_sfpu_operation_quasar_impl<OPERATION, DST_SYNC, is_fp32_dest_acc_en, DataFormat::Float16_b, APPROX, ITERATIONS>(
            in0_tile, in1_tile, in2_tile, out_tile, vector_mode, scalar);
    }
    else
    {
        LLK_ASSERT(false, "Unsupported Quasar ternary SFPU format");
    }
}

} // namespace test_utils
