// SPDX-License-Identifier: Apache-2.0
//
// llk_analysis coverage gap round (ops/blackhole/sfpu_gaps), with the one 32-row call forms (<op>_one) of the int32
// division, gcd and lcm entry points added for the #58743 check of ops/blackhole/sfpu_int_typecast: dispatch of the Blackhole SFPU compute API entry points
// that no checked-in perf module times. Each branch reproduces the body, the template arguments, the vector mode and
// the init of the corresponding <op>_tile / <op>_tile_init in tt_metal/hw/inc/api/compute as of the counter base,
// with the scalar arguments ttnn passes (threshold 0.5f or 5, shift 3, exponent 3, bit mask 0x0F0F0F0F). Shadow tree
// only; nothing here is part of the harness.

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_sfpu_types.h"

// The compute API's name for the kernel's dest mode; the init macros expand it.
#define DST_ACCUM_MODE is_fp32_dest_acc_en

#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_binop_with_unary.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_div_int32.h"
#include "llk_sfpu/ckernel_sfpu_div_int32_floor.h"
#include "llk_sfpu/ckernel_sfpu_binary_remainder.h"
#include "llk_sfpu/ckernel_sfpu_binary_fmod.h"
#include "llk_sfpu/ckernel_sfpu_gcd.h"
#include "llk_sfpu/ckernel_sfpu_identity.h"
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/ckernel_sfpu_isclose.h"
#include "llk_sfpu/ckernel_sfpu_lcm.h"
#include "llk_sfpu/ckernel_sfpu_lgamma.h"
#include "llk_sfpu/ckernel_sfpu_logical_not.h"
#include "llk_sfpu/ckernel_sfpu_logsigmoid.h"
#include "llk_sfpu/ckernel_sfpu_mac.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_max_pool_indices.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "llk_sfpu/ckernel_sfpu_relu.h"
#include "llk_sfpu/ckernel_sfpu_remainder.h"
#include "llk_sfpu/ckernel_sfpu_rsub_int32.h"
#include "llk_sfpu/ckernel_sfpu_shift.h"
#include "llk_sfpu/ckernel_sfpu_signbit.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"
#include "llk_sfpu/ckernel_sfpu_unary_comp.h"
#include "llk_sfpu/ckernel_sfpu_unary_max_min.h"
#include "llk_sfpu/ckernel_sfpu_unary_power.h"
#include "llk_sfpu/ckernel_sfpu_unary_shift.h"
#include "sfpu/ckernel_sfpu_add_int.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "sfpu/ckernel_sfpu_fill.h"
#include "sfpu/ckernel_sfpu_isinf_isnan.h"
#include "sfpu/ckernel_sfpu_mul_int.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "sfpu/ckernel_sfpu_sub_int.h"

namespace ckernel
{
namespace gaps_x
{

enum class GapOp
{
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
    unary_ne,
    unary_eq,
    unary_ne_int32,
    unary_eq_int32,
    unary_gt_int32,
    unary_ge_int32,
    unary_lt_int32,
    unary_le_int32,
    gtz_int32,
    nez_int32,
    gez_int32,
    ltz_int32,
    eqz_int32,
    lez_int32,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    left_shift,
    right_shift,
    relu_max_int32,
    relu_min_int32,
    relu_max_uint32,
    relu_max_uint16,
    relu_min_uint32,
    relu_min_uint16,
    relu_int32,
    identity_uint32,
    clamp_int32,
    stochastic_round,
    signbit_int32,
    tiled_prod,
    power_iterative,
    alt_complex_rotate90,
    unary_max_int32,
    unary_min_int32,
    unary_max_uint32,
    unary_min_uint32,
    sum_int_col,
    sum_int_row,
    add_int_unary,
    rsub_unary_int32,
    add_unary_int32,
    sub_unary_int32,
    fill_int,
    fill_bitcast,
    logical_not,
    negative_int32,
    remainder_uint32,
    mask_posinf,
    lgamma_stirling_float,
    gcd,
    lcm,
    isclose,
    logsigmoid,
    mul_int_uint16,
    add_int_tile,
    sub_int_tile,
    rsub_int_tile,
    binary_left_shift,
    binary_right_shift,
    binary_logical_right_shift,
    div_int32,
    div_int32_floor,
    div_int32_trunc,
    remainder_int32_bin,
    remainder_uint32_bin,
    fmod_int32_bin,
    nextafter,
    nextafter_bf16,
    max_reduce_with_indices,
    lgamma_adjusted,
    mac,
    gcd_one,
    lcm_one,
    div_int32_one,
    div_int32_floor_one,
    div_int32_trunc_one,
    remainder_int32_bin_one,
    remainder_uint32_bin_one,
    fmod_int32_bin_one,
};

// Scalar arguments as ttnn passes them (bit patterns of floats, plain integers for the integer entry points).
constexpr std::uint32_t F_HALF    = 0x3F000000u; // 0.5f
constexpr std::uint32_t F_ONE     = 0x3F800000u; // 1.0f
constexpr std::uint32_t I_FIVE    = 5u;
constexpr std::uint32_t I_NEG5    = 0xFFFFFFFBu;
constexpr std::uint32_t BITMASK   = 0x0F0F0F0Fu;
constexpr std::uint32_t SHIFT     = 3u;
constexpr std::uint32_t EXP3      = 3u;
constexpr std::uint32_t RTOL_BITS = 0x3727C5ACu; // 1e-5f
constexpr std::uint32_t ATOL_BITS = 0x322BCC77u; // 1e-8f
constexpr std::uint32_t FILL_INT  = 7u;
constexpr std::uint32_t DIVISOR   = 7u;

template <DataFormat FMT>
constexpr InstrModLoadStore int_mode()
{
    return (FMT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
}

template <DataFormat FMT>
constexpr InstrModLoadStore logical_not_mode()
{
    return (FMT == DataFormat::Float32 || FMT == DataFormat::Float16_b || FMT == DataFormat::Bfp8_b || FMT == DataFormat::Bfp4_b)
               ? InstrModLoadStore::DEFAULT
           : (FMT == DataFormat::UInt16) ? InstrModLoadStore::LO16
                                         : InstrModLoadStore::INT32;
}

template <GapOp OP, bool APPROX, DataFormat FMT>
inline void gaps_init()
{
    if constexpr (OP == GapOp::isinf)
    {
        SFPU_UNARY_INIT(isinf);
    }
    else if constexpr (OP == GapOp::isposinf)
    {
        SFPU_UNARY_INIT(isposinf);
    }
    else if constexpr (OP == GapOp::isneginf)
    {
        SFPU_UNARY_INIT(isneginf);
    }
    else if constexpr (OP == GapOp::isnan)
    {
        SFPU_UNARY_INIT(isnan);
    }
    else if constexpr (OP == GapOp::isfinite)
    {
        SFPU_UNARY_INIT(isfinite);
    }
    else if constexpr (OP == GapOp::unary_ne || OP == GapOp::unary_ne_int32)
    {
        SFPU_UNARY_INIT(unary_ne);
    }
    else if constexpr (OP == GapOp::unary_eq || OP == GapOp::unary_eq_int32)
    {
        SFPU_UNARY_INIT(unary_eq);
    }
    else if constexpr (OP == GapOp::unary_gt_int32)
    {
        SFPU_UNARY_INIT(unary_gt);
    }
    else if constexpr (OP == GapOp::unary_ge_int32)
    {
        SFPU_UNARY_INIT(unary_ge);
    }
    else if constexpr (OP == GapOp::unary_lt_int32)
    {
        SFPU_UNARY_INIT(unary_lt);
    }
    else if constexpr (OP == GapOp::unary_le_int32)
    {
        SFPU_UNARY_INIT(unary_le);
    }
    else if constexpr (OP == GapOp::gtz_int32)
    {
        SFPU_UNARY_INIT(greater_than_zero);
    }
    else if constexpr (OP == GapOp::nez_int32)
    {
        SFPU_UNARY_INIT(not_equal_zero);
    }
    else if constexpr (OP == GapOp::gez_int32)
    {
        SFPU_UNARY_INIT(greater_than_equal_zero);
    }
    else if constexpr (OP == GapOp::ltz_int32)
    {
        SFPU_UNARY_INIT(less_than_zero);
    }
    else if constexpr (OP == GapOp::eqz_int32)
    {
        SFPU_UNARY_INIT(equal_zero);
    }
    else if constexpr (OP == GapOp::lez_int32)
    {
        SFPU_UNARY_INIT(less_than_equal_zero);
    }
    else if constexpr (OP == GapOp::bitwise_and)
    {
        SFPU_UNARY_INIT(bitwise_and);
    }
    else if constexpr (OP == GapOp::bitwise_or)
    {
        SFPU_UNARY_INIT(bitwise_or);
    }
    else if constexpr (OP == GapOp::bitwise_xor)
    {
        SFPU_UNARY_INIT(bitwise_xor);
    }
    else if constexpr (OP == GapOp::left_shift)
    {
        SFPU_UNARY_INIT(left_shift);
    }
    else if constexpr (OP == GapOp::right_shift)
    {
        SFPU_UNARY_INIT(right_shift);
    }
    else if constexpr (OP == GapOp::relu_max_int32 || OP == GapOp::relu_max_uint32 || OP == GapOp::relu_max_uint16)
    {
        SFPU_UNARY_INIT(relu_max);
    }
    else if constexpr (OP == GapOp::relu_min_int32 || OP == GapOp::relu_min_uint32 || OP == GapOp::relu_min_uint16 || OP == GapOp::relu_int32)
    {
        SFPU_UNARY_INIT(relu_min);
    }
    else if constexpr (OP == GapOp::identity_uint32 || OP == GapOp::stochastic_round || OP == GapOp::rsub_unary_int32 || OP == GapOp::add_unary_int32 || OP == GapOp::sub_unary_int32)
    {
        SFPU_UNARY_INIT(unused);
    }
    else if constexpr (OP == GapOp::clamp_int32)
    {
        SFPU_UNARY_INIT(clamp);
    }
    else if constexpr (OP == GapOp::signbit_int32)
    {
        llk_math_eltwise_unary_sfpu_init<SfpuType::signbit>(ckernel::sfpu::signbit_int32_init);
    }
    else if constexpr (OP == GapOp::tiled_prod)
    {
        SFPU_UNARY_INIT(tiled_prod);
    }
    else if constexpr (OP == GapOp::power_iterative)
    {
        SFPU_UNARY_INIT(power);
    }
    else if constexpr (OP == GapOp::alt_complex_rotate90)
    {
        SFPU_UNARY_INIT(alt_complex_rotate90);
    }
    else if constexpr (OP == GapOp::unary_max_int32)
    {
        SFPU_UNARY_INIT_FN(unary_max_int32, sfpu::unary_max_min_int32_init, (true, false));
    }
    else if constexpr (OP == GapOp::unary_min_int32)
    {
        SFPU_UNARY_INIT_FN(unary_min_int32, sfpu::unary_max_min_int32_init, (false, false));
    }
    else if constexpr (OP == GapOp::unary_max_uint32)
    {
        SFPU_UNARY_INIT_FN(unary_max_uint32, sfpu::unary_max_min_int32_init, (true, true));
    }
    else if constexpr (OP == GapOp::unary_min_uint32)
    {
        SFPU_UNARY_INIT_FN(unary_min_uint32, sfpu::unary_max_min_int32_init, (false, true));
    }
    else if constexpr (OP == GapOp::sum_int_col || OP == GapOp::sum_int_row || OP == GapOp::add_int_unary)
    {
        SFPU_UNARY_INIT_FN(unused, sfpu::sum_int_init, (APPROX));
    }
    else if constexpr (OP == GapOp::fill_int || OP == GapOp::fill_bitcast)
    {
        SFPU_UNARY_INIT(fill);
    }
    else if constexpr (OP == GapOp::logical_not)
    {
        SFPU_UNARY_INIT(logical_not_unary);
    }
    else if constexpr (OP == GapOp::negative_int32)
    {
        SFPU_UNARY_INIT(negative);
    }
    else if constexpr (OP == GapOp::remainder_uint32)
    {
        SFPU_UNARY_INIT_FN(remainder_uint32, sfpu::remainder_uint32_init, (APPROX));
    }
    else if constexpr (OP == GapOp::mask_posinf)
    {
        SFPU_UNARY_INIT(mask);
    }
    else if constexpr (OP == GapOp::lgamma_stirling_float)
    {
        SFPU_BINARY_INIT_FN(lgamma, sfpu::lgamma_stirling_init, (APPROX));
    }
    else if constexpr (OP == GapOp::gcd || OP == GapOp::gcd_one)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(gcd, sfpu::calculate_sfpu_gcd_init);
    }
    else if constexpr (OP == GapOp::lcm || OP == GapOp::lcm_one)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(lcm, sfpu::calculate_sfpu_lcm_init);
    }
    else if constexpr (OP == GapOp::isclose)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(isclose, sfpu::isclose_init);
    }
    else if constexpr (
        OP == GapOp::logsigmoid || OP == GapOp::add_int_tile || OP == GapOp::sub_int_tile || OP == GapOp::rsub_int_tile || OP == GapOp::binary_left_shift ||
        OP == GapOp::binary_right_shift || OP == GapOp::binary_logical_right_shift)
    {
        SFPU_BINARY_INIT(unused);
    }
    else if constexpr (OP == GapOp::mul_int_uint16)
    {
        SFPU_BINARY_INIT_FN(mul_uint16, sfpu::_init_mul_int_, (APPROX));
    }
    else if constexpr (OP == GapOp::div_int32 || OP == GapOp::div_int32_one)
    {
        SFPU_BINARY_INIT_FN(div_int32, sfpu::div_init, (APPROX));
    }
    else if constexpr (OP == GapOp::div_int32_floor || OP == GapOp::div_int32_floor_one)
    {
        SFPU_BINARY_INIT_FN(div_int32_floor, sfpu::div_floor_init, (APPROX));
    }
    else if constexpr (OP == GapOp::div_int32_trunc || OP == GapOp::div_int32_trunc_one)
    {
        SFPU_BINARY_INIT_FN(div_int32_trunc, sfpu::div_trunc_init, (APPROX));
    }
    else if constexpr (OP == GapOp::remainder_int32_bin || OP == GapOp::remainder_int32_bin_one)
    {
        SFPU_BINARY_INIT_FN(remainder_int32, sfpu::remainder_int32_init, (APPROX));
    }
    else if constexpr (OP == GapOp::remainder_uint32_bin || OP == GapOp::remainder_uint32_bin_one)
    {
        SFPU_BINARY_INIT_FN(remainder_uint32, sfpu::remainder_uint32_init, (APPROX));
    }
    else if constexpr (OP == GapOp::fmod_int32_bin || OP == GapOp::fmod_int32_bin_one)
    {
        SFPU_BINARY_INIT_FN(fmod_int32, sfpu::fmod_int32_init, (APPROX));
    }
    else if constexpr (OP == GapOp::nextafter)
    {
        SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::NEXTAFTER));
    }
    else if constexpr (OP == GapOp::nextafter_bf16)
    {
        SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::NEXTAFTER_BF16));
    }
    else if constexpr (OP == GapOp::max_reduce_with_indices)
    {
        SFPU_BINARY_INIT_FN(max_pool_with_indices, sfpu::init_max_pool_with_indices, (true, ckernel::DataLayout::TILE));
    }
    else if constexpr (OP == GapOp::lgamma_adjusted)
    {
        SFPU_TERNARY_INIT(lgamma);
    }
    else if constexpr (OP == GapOp::mac)
    {
        SFPU_TERNARY_INIT_FN(mac, sfpu::mac_init, (APPROX, is_fp32_dest_acc_en, FMT));
    }
    else
    {
        static_assert(OP == GapOp::isinf, "gaps_init: no branch for this op");
    }
}

// t0 is the tile the carrier copied; t1, t2, t3 are the next tiles of the DEST half (operands of the two- and
// three-input entry points, as the checked-in binary perf kernel places them).
template <GapOp OP, bool APPROX, DataFormat FMT>
inline void gaps_call(std::uint32_t t0, std::uint32_t t1, std::uint32_t t2, std::uint32_t t3)
{
    constexpr DstSync DST_SYNC_MODE = DstSync::SyncHalf;
    if constexpr (OP == GapOp::isinf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isinf, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::isposinf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isposinf, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::isneginf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isneginf, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::isnan)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isnan, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::isfinite)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isfinite, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::unary_ne)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_ne, (APPROX, 8), t0, VectorMode::RC, F_HALF);
    }
    else if constexpr (OP == GapOp::unary_eq)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_eq, (APPROX, 8), t0, VectorMode::RC, F_HALF);
    }
    else if constexpr (OP == GapOp::unary_ne_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_unary_int, (APPROX, SfpuType::unary_ne, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_eq_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_unary_int, (APPROX, SfpuType::unary_eq, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_gt_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_gt, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_ge_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_ge, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_lt_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_lt, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_le_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_le, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::gtz_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::greater_than_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::nez_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::not_equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::gez_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::greater_than_equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::ltz_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::less_than_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::eqz_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::lez_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::less_than_equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::bitwise_and)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_unary_bitwise, (APPROX, sfpu::UnaryBitwiseOp::AND, FMT), t0, VectorMode::RC, BITMASK);
    }
    else if constexpr (OP == GapOp::bitwise_or)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_unary_bitwise, (APPROX, sfpu::UnaryBitwiseOp::OR, FMT), t0, VectorMode::RC, BITMASK);
    }
    else if constexpr (OP == GapOp::bitwise_xor)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_unary_bitwise, (APPROX, sfpu::UnaryBitwiseOp::XOR, FMT), t0, VectorMode::RC, BITMASK);
    }
    else if constexpr (OP == GapOp::left_shift)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_left_shift, (APPROX, FMT), t0, VectorMode::RC, SHIFT);
    }
    else if constexpr (OP == GapOp::right_shift)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_right_shift, (APPROX, FMT), t0, VectorMode::RC, SHIFT);
    }
    else if constexpr (OP == GapOp::relu_max_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_int, (APPROX, false, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::relu_min_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_int, (APPROX, true, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::relu_max_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, false, DataFormat::UInt32, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::relu_max_uint16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, false, DataFormat::UInt16, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::relu_min_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, true, DataFormat::UInt32, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::relu_min_uint16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, true, DataFormat::UInt16, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::relu_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _relu_min_, (sfpi::vInt, APPROX, 8, uint32_t), t0, VectorMode::RC, 0u);
    }
    else if constexpr (OP == GapOp::identity_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_identity_uint, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::clamp_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_clamp_int32, (APPROX, 8), t0, VectorMode::RC, I_NEG5, I_FIVE);
    }
    else if constexpr (OP == GapOp::stochastic_round)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_stochastic_round_, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::signbit_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_signbit_int32, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::tiled_prod)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tiled_prod, (APPROX), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::power_iterative)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_power_iterative, (APPROX, 8), t0, VectorMode::RC, EXP3);
    }
    else if constexpr (OP == GapOp::alt_complex_rotate90)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (APPROX), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::unary_max_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (true, false, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_min_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (false, false, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_max_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (true, true, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::unary_min_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (false, true, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::sum_int_col)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sum_int_col, (APPROX), t0, VectorMode::R);
    }
    else if constexpr (OP == GapOp::sum_int_row)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sum_int_row, (APPROX), t0, VectorMode::C);
    }
    else if constexpr (OP == GapOp::add_int_unary)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, add_int, (APPROX, 8), t0, VectorMode::RC, 2u);
    }
    else if constexpr (OP == GapOp::rsub_unary_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rsub_scalar_int32, (APPROX, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::add_unary_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_add_int32, (APPROX, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::sub_unary_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sub_int32, (APPROX, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == GapOp::fill_int)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_int_, (APPROX, int_mode<FMT>(), 8), t0, VectorMode::RC, FILL_INT);
    }
    else if constexpr (OP == GapOp::fill_bitcast)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_bitcast_, (APPROX, 8), t0, VectorMode::RC, F_ONE);
    }
    else if constexpr (OP == GapOp::logical_not)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, logical_not_mode<FMT>(), 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::negative_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_negative_int_, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::remainder_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_remainder_uint32_scalar, (APPROX, 8), t0, VectorMode::RC, DIVISOR);
    }
    else if constexpr (OP == GapOp::mask_posinf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_mask_posinf, (true), t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::lgamma_stirling_float)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_stirling_fp32, (APPROX), t0, t1, t2, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::gcd)
    {
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_gcd, t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::lcm)
    {
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_lcm, t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::isclose)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_isclose, (APPROX, 8, false), t0, t1, t0, VectorMode::RC, RTOL_BITS, ATOL_BITS);
    }
    else if constexpr (OP == GapOp::logsigmoid)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logsigmoid, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::mul_int_uint16)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _mul_int_, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::add_int_tile)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _add_int_, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::sub_int_tile)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _sub_int_, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::rsub_int_tile)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rsub_int, (APPROX, int_mode<FMT>(), 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::binary_left_shift)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_binary_left_shift, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::binary_right_shift)
    {
        if constexpr (FMT == DataFormat::UInt32)
        {
            SFPU_BINARY_CALL(
                DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_clamped_logical_right_shift, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
        }
        else
        {
            SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_binary_right_shift, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
        }
    }
    else if constexpr (OP == GapOp::binary_logical_right_shift)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logical_right_shift, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::div_int32)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::div_int32_floor)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32_floor, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::div_int32_trunc)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32_trunc, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::remainder_int32_bin)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_remainder_int32, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::remainder_uint32_bin)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_remainder_uint32, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::fmod_int32_bin)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_fmod_int32, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::gcd_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_gcd, (32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::lcm_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_lcm, (32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::div_int32_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32, (APPROX, 32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::div_int32_floor_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32_floor, (APPROX, 32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::div_int32_trunc_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32_trunc, (APPROX, 32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::remainder_int32_bin_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_remainder_int32, (APPROX, 32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::remainder_uint32_bin_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_remainder_uint32, (APPROX, 32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::fmod_int32_bin_one)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_fmod_int32, (APPROX, 32), t0, t1, t0, VectorMode::None);
    }
    else if constexpr (OP == GapOp::nextafter)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_binary, (APPROX, BinaryOp::NEXTAFTER, 8, is_fp32_dest_acc_en), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::nextafter_bf16)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_binary, (APPROX, BinaryOp::NEXTAFTER_BF16, 8, is_fp32_dest_acc_en), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::max_reduce_with_indices)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_max_pool_with_indices,
            (true, is_fp32_dest_acc_en, 9, 8, ckernel::DataLayout::TILE, false),
            t0,
            t1,
            0,
            VectorMode::RC,
            0u);
    }
    else if constexpr (OP == GapOp::lgamma_adjusted)
    {
        SFPU_TERNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_adjusted, (APPROX, is_fp32_dest_acc_en), t0, t1, t2, t3, VectorMode::RC);
    }
    else if constexpr (OP == GapOp::mac)
    {
        SFPU_TERNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_mac, (APPROX, is_fp32_dest_acc_en, FMT, 8), t0, t1, t2, t0, VectorMode::RC);
    }
    else
    {
        static_assert(OP == GapOp::isinf, "gaps_call: no branch for this op");
    }
}

} // namespace gaps_x
} // namespace ckernel
