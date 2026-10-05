// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The compute API SFPU entry points the registry sweeps do not time, at the llk layer: each branch is the body,
// template arguments, vector mode and init of its <op>_tile / <op>_tile_init in tt_metal/hw/inc/api/compute, with the
// scalar arguments ttnn passes.

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_sfpu_types.h"

// The compute API's name for the kernel's dest mode; the init macros expand it.
#define DST_ACCUM_MODE is_fp32_dest_acc_en

// clang-format off
// The macro headers declare the init functions, so they come before the bodies that define them.
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
// clang-format on

#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_binop_with_unary.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_clamp.h"
#include "llk_sfpu/ckernel_sfpu_comp.h"
#include "llk_sfpu/ckernel_sfpu_div_int32.h"
#include "llk_sfpu/ckernel_sfpu_div_int32_floor.h"
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
namespace compute_api_perf
{

enum class ApiOp
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
    nextafter,
    nextafter_bf16,
    max_reduce_with_indices,
    lgamma_adjusted,
    mac,
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
    return (FMT == DataFormat::Float32 || FMT == DataFormat::Float16_b || FMT == DataFormat::Bfp8_b || FMT == DataFormat::Bfp4_b) ? InstrModLoadStore::DEFAULT
           : (FMT == DataFormat::UInt16)                                                                                          ? InstrModLoadStore::LO16
                                                                                                                                  : InstrModLoadStore::INT32;
}

template <ApiOp OP, bool APPROX, DataFormat FMT>
inline void api_init()
{
    if constexpr (OP == ApiOp::isinf)
    {
        SFPU_UNARY_INIT(isinf);
    }
    else if constexpr (OP == ApiOp::isposinf)
    {
        SFPU_UNARY_INIT(isposinf);
    }
    else if constexpr (OP == ApiOp::isneginf)
    {
        SFPU_UNARY_INIT(isneginf);
    }
    else if constexpr (OP == ApiOp::isnan)
    {
        SFPU_UNARY_INIT(isnan);
    }
    else if constexpr (OP == ApiOp::isfinite)
    {
        SFPU_UNARY_INIT(isfinite);
    }
    else if constexpr (OP == ApiOp::unary_ne || OP == ApiOp::unary_ne_int32)
    {
        SFPU_UNARY_INIT(unary_ne);
    }
    else if constexpr (OP == ApiOp::unary_eq || OP == ApiOp::unary_eq_int32)
    {
        SFPU_UNARY_INIT(unary_eq);
    }
    else if constexpr (OP == ApiOp::unary_gt_int32)
    {
        SFPU_UNARY_INIT(unary_gt);
    }
    else if constexpr (OP == ApiOp::unary_ge_int32)
    {
        SFPU_UNARY_INIT(unary_ge);
    }
    else if constexpr (OP == ApiOp::unary_lt_int32)
    {
        SFPU_UNARY_INIT(unary_lt);
    }
    else if constexpr (OP == ApiOp::unary_le_int32)
    {
        SFPU_UNARY_INIT(unary_le);
    }
    else if constexpr (OP == ApiOp::gtz_int32)
    {
        SFPU_UNARY_INIT(greater_than_zero);
    }
    else if constexpr (OP == ApiOp::nez_int32)
    {
        SFPU_UNARY_INIT(not_equal_zero);
    }
    else if constexpr (OP == ApiOp::gez_int32)
    {
        SFPU_UNARY_INIT(greater_than_equal_zero);
    }
    else if constexpr (OP == ApiOp::ltz_int32)
    {
        SFPU_UNARY_INIT(less_than_zero);
    }
    else if constexpr (OP == ApiOp::eqz_int32)
    {
        SFPU_UNARY_INIT(equal_zero);
    }
    else if constexpr (OP == ApiOp::lez_int32)
    {
        SFPU_UNARY_INIT(less_than_equal_zero);
    }
    else if constexpr (OP == ApiOp::bitwise_and)
    {
        SFPU_UNARY_INIT(bitwise_and);
    }
    else if constexpr (OP == ApiOp::bitwise_or)
    {
        SFPU_UNARY_INIT(bitwise_or);
    }
    else if constexpr (OP == ApiOp::bitwise_xor)
    {
        SFPU_UNARY_INIT(bitwise_xor);
    }
    else if constexpr (OP == ApiOp::left_shift)
    {
        SFPU_UNARY_INIT(left_shift);
    }
    else if constexpr (OP == ApiOp::right_shift)
    {
        SFPU_UNARY_INIT(right_shift);
    }
    else if constexpr (OP == ApiOp::relu_max_int32 || OP == ApiOp::relu_max_uint32 || OP == ApiOp::relu_max_uint16)
    {
        SFPU_UNARY_INIT(relu_max);
    }
    else if constexpr (OP == ApiOp::relu_min_int32 || OP == ApiOp::relu_min_uint32 || OP == ApiOp::relu_min_uint16 || OP == ApiOp::relu_int32)
    {
        SFPU_UNARY_INIT(relu_min);
    }
    else if constexpr (
        OP == ApiOp::identity_uint32 || OP == ApiOp::stochastic_round || OP == ApiOp::rsub_unary_int32 || OP == ApiOp::add_unary_int32 ||
        OP == ApiOp::sub_unary_int32)
    {
        SFPU_UNARY_INIT(unused);
    }
    else if constexpr (OP == ApiOp::clamp_int32)
    {
        SFPU_UNARY_INIT(clamp);
    }
    else if constexpr (OP == ApiOp::signbit_int32)
    {
        llk_math_eltwise_unary_sfpu_init<SfpuType::signbit>(ckernel::sfpu::signbit_int32_init);
    }
    else if constexpr (OP == ApiOp::tiled_prod)
    {
        SFPU_UNARY_INIT(tiled_prod);
    }
    else if constexpr (OP == ApiOp::power_iterative)
    {
        SFPU_UNARY_INIT(power);
    }
    else if constexpr (OP == ApiOp::alt_complex_rotate90)
    {
        SFPU_UNARY_INIT(alt_complex_rotate90);
    }
    else if constexpr (OP == ApiOp::unary_max_int32)
    {
        SFPU_UNARY_INIT_FN(unary_max_int32, sfpu::unary_max_min_int32_init, (true, false));
    }
    else if constexpr (OP == ApiOp::unary_min_int32)
    {
        SFPU_UNARY_INIT_FN(unary_min_int32, sfpu::unary_max_min_int32_init, (false, false));
    }
    else if constexpr (OP == ApiOp::unary_max_uint32)
    {
        SFPU_UNARY_INIT_FN(unary_max_uint32, sfpu::unary_max_min_int32_init, (true, true));
    }
    else if constexpr (OP == ApiOp::unary_min_uint32)
    {
        SFPU_UNARY_INIT_FN(unary_min_uint32, sfpu::unary_max_min_int32_init, (false, true));
    }
    else if constexpr (OP == ApiOp::sum_int_col || OP == ApiOp::sum_int_row || OP == ApiOp::add_int_unary)
    {
        SFPU_UNARY_INIT_FN(unused, sfpu::sum_int_init, (APPROX));
    }
    else if constexpr (OP == ApiOp::fill_int || OP == ApiOp::fill_bitcast)
    {
        SFPU_UNARY_INIT(fill);
    }
    else if constexpr (OP == ApiOp::logical_not)
    {
        SFPU_UNARY_INIT(logical_not_unary);
    }
    else if constexpr (OP == ApiOp::negative_int32)
    {
        SFPU_UNARY_INIT(negative);
    }
    else if constexpr (OP == ApiOp::remainder_uint32)
    {
        SFPU_UNARY_INIT_FN(remainder_uint32, sfpu::remainder_uint32_init, (APPROX));
    }
    else if constexpr (OP == ApiOp::mask_posinf)
    {
        SFPU_UNARY_INIT(mask);
    }
    else if constexpr (OP == ApiOp::lgamma_stirling_float)
    {
        SFPU_BINARY_INIT_FN(lgamma, sfpu::lgamma_stirling_init, (APPROX));
    }
    else if constexpr (OP == ApiOp::gcd)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(gcd, sfpu::calculate_sfpu_gcd_init);
    }
    else if constexpr (OP == ApiOp::lcm)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(lcm, sfpu::calculate_sfpu_lcm_init);
    }
    else if constexpr (OP == ApiOp::isclose)
    {
        SFPU_BINARY_INIT_FN_NO_ARGS(isclose, sfpu::isclose_init);
    }
    else if constexpr (
        OP == ApiOp::logsigmoid || OP == ApiOp::add_int_tile || OP == ApiOp::sub_int_tile || OP == ApiOp::rsub_int_tile || OP == ApiOp::binary_left_shift ||
        OP == ApiOp::binary_right_shift || OP == ApiOp::binary_logical_right_shift)
    {
        SFPU_BINARY_INIT(unused);
    }
    else if constexpr (OP == ApiOp::mul_int_uint16)
    {
        SFPU_BINARY_INIT_FN(mul_uint16, sfpu::_init_mul_int_, (APPROX));
    }
    else if constexpr (OP == ApiOp::div_int32)
    {
        SFPU_BINARY_INIT_FN(div_int32, sfpu::div_init, (APPROX));
    }
    else if constexpr (OP == ApiOp::div_int32_floor)
    {
        SFPU_BINARY_INIT_FN(div_int32_floor, sfpu::div_floor_init, (APPROX));
    }
    else if constexpr (OP == ApiOp::div_int32_trunc)
    {
        SFPU_BINARY_INIT_FN(div_int32_trunc, sfpu::div_trunc_init, (APPROX));
    }
    else if constexpr (OP == ApiOp::nextafter)
    {
        SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::NEXTAFTER));
    }
    else if constexpr (OP == ApiOp::nextafter_bf16)
    {
        SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::NEXTAFTER_BF16));
    }
    else if constexpr (OP == ApiOp::max_reduce_with_indices)
    {
        SFPU_BINARY_INIT_FN(max_pool_with_indices, sfpu::init_max_pool_with_indices, (true, ckernel::DataLayout::ROW_MAJOR));
    }
    else if constexpr (OP == ApiOp::lgamma_adjusted)
    {
        SFPU_TERNARY_INIT(lgamma);
    }
    else if constexpr (OP == ApiOp::mac)
    {
        SFPU_TERNARY_INIT_FN(mac, sfpu::mac_init, (APPROX, is_fp32_dest_acc_en, FMT));
    }
    else
    {
        static_assert(OP == ApiOp::isinf, "api_init: no branch for this op");
    }
}

// t0 is the tile the carrier copied; t1 and t2 are the next tiles of the DEST half (operands of the two- and
// three-input entry points, as the checked-in binary perf kernel places them).
template <ApiOp OP, bool APPROX, DataFormat FMT>
inline void api_call(std::uint32_t t0, std::uint32_t t1, std::uint32_t t2)
{
    constexpr DstSync DST_SYNC_MODE = DstSync::SyncHalf;
    if constexpr (OP == ApiOp::isinf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isinf, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::isposinf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isposinf, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::isneginf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isneginf, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::isnan)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isnan, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::isfinite)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_sfpu_isinf_isnan_, (SfpuType::isfinite, APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::unary_ne)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_ne, (APPROX, 8), t0, VectorMode::RC, F_HALF);
    }
    else if constexpr (OP == ApiOp::unary_eq)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_eq, (APPROX, 8), t0, VectorMode::RC, F_HALF);
    }
    else if constexpr (OP == ApiOp::unary_ne_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_unary_int, (APPROX, SfpuType::unary_ne, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_eq_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_unary_int, (APPROX, SfpuType::unary_eq, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_gt_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_gt, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_ge_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_ge, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_lt_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_lt, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_le_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, SfpuType::unary_le, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::gtz_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::greater_than_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::nez_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::not_equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::gez_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::greater_than_equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::ltz_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::less_than_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::eqz_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::lez_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_comp_int, (APPROX, SfpuType::less_than_equal_zero), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::bitwise_and)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_unary_bitwise, (APPROX, sfpu::UnaryBitwiseOp::AND, FMT), t0, VectorMode::RC, BITMASK);
    }
    else if constexpr (OP == ApiOp::bitwise_or)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_unary_bitwise, (APPROX, sfpu::UnaryBitwiseOp::OR, FMT), t0, VectorMode::RC, BITMASK);
    }
    else if constexpr (OP == ApiOp::bitwise_xor)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_unary_bitwise, (APPROX, sfpu::UnaryBitwiseOp::XOR, FMT), t0, VectorMode::RC, BITMASK);
    }
    else if constexpr (OP == ApiOp::left_shift)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_left_shift, (APPROX, FMT), t0, VectorMode::RC, SHIFT);
    }
    else if constexpr (OP == ApiOp::right_shift)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_right_shift, (APPROX, FMT), t0, VectorMode::RC, SHIFT);
    }
    else if constexpr (OP == ApiOp::relu_max_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_int, (APPROX, false, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::relu_min_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_int, (APPROX, true, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::relu_max_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, false, DataFormat::UInt32, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::relu_max_uint16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, false, DataFormat::UInt16, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::relu_min_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, true, DataFormat::UInt32, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::relu_min_uint16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, relu_clamp_uint, (APPROX, true, DataFormat::UInt16, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::relu_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _relu_min_, (sfpi::vInt, APPROX, 8, std::uint32_t), t0, VectorMode::RC, 0u);
    }
    else if constexpr (OP == ApiOp::identity_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_identity_uint, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::clamp_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_clamp_int32, (APPROX, 8), t0, VectorMode::RC, I_NEG5, I_FIVE);
    }
    else if constexpr (OP == ApiOp::stochastic_round)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_stochastic_round_, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::signbit_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_signbit_int32, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::tiled_prod)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tiled_prod, (APPROX), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::power_iterative)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_power_iterative, (APPROX, 8), t0, VectorMode::RC, EXP3);
    }
    else if constexpr (OP == ApiOp::alt_complex_rotate90)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (APPROX), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::unary_max_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (true, false, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_min_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (false, false, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_max_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (true, true, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::unary_min_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_max_min_int32, (false, true, APPROX), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::sum_int_col)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sum_int_col, (APPROX), t0, VectorMode::R);
    }
    else if constexpr (OP == ApiOp::sum_int_row)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sum_int_row, (APPROX), t0, VectorMode::C);
    }
    else if constexpr (OP == ApiOp::add_int_unary)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, add_int, (APPROX, 8), t0, VectorMode::RC, 2u);
    }
    else if constexpr (OP == ApiOp::rsub_unary_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rsub_scalar_int32, (APPROX, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::add_unary_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_add_int32, (APPROX, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::sub_unary_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sub_int32, (APPROX, 8), t0, VectorMode::RC, I_FIVE);
    }
    else if constexpr (OP == ApiOp::fill_int)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_int_, (APPROX, int_mode<FMT>(), 8), t0, VectorMode::RC, FILL_INT);
    }
    else if constexpr (OP == ApiOp::fill_bitcast)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_fill_bitcast_, (APPROX, 8), t0, VectorMode::RC, F_ONE);
    }
    else if constexpr (OP == ApiOp::logical_not)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logical_not, (APPROX, logical_not_mode<FMT>(), 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::negative_int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_negative_int_, (APPROX, 8), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::remainder_uint32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_remainder_uint32_scalar, (APPROX, 8), t0, VectorMode::RC, DIVISOR);
    }
    else if constexpr (OP == ApiOp::mask_posinf)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_mask_posinf, (true), t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::lgamma_stirling_float)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_stirling_fp32, (APPROX), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::gcd)
    {
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_gcd, t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::lcm)
    {
        SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_lcm, t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::isclose)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_isclose, (APPROX, 8, false), t0, t1, t0, VectorMode::RC, RTOL_BITS, ATOL_BITS);
    }
    else if constexpr (OP == ApiOp::logsigmoid)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logsigmoid, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::mul_int_uint16)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _mul_int_, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::add_int_tile)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _add_int_, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::sub_int_tile)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _sub_int_, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::rsub_int_tile)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rsub_int, (APPROX, int_mode<FMT>(), 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::binary_left_shift)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_binary_left_shift, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::binary_right_shift)
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
    else if constexpr (OP == ApiOp::binary_logical_right_shift)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logical_right_shift, (APPROX, 8, int_mode<FMT>(), false), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::div_int32)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::div_int32_floor)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32_floor, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::div_int32_trunc)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_div_int32_trunc, (APPROX, 8), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::nextafter)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_binary, (APPROX, BinaryOp::NEXTAFTER, 8, is_fp32_dest_acc_en), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::nextafter_bf16)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_binary, (APPROX, BinaryOp::NEXTAFTER_BF16, 8, is_fp32_dest_acc_en), t0, t1, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::max_reduce_with_indices)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_max_pool_with_indices,
            (true, is_fp32_dest_acc_en, 9, 8, ckernel::DataLayout::ROW_MAJOR, false),
            t0,
            t1,
            0,
            VectorMode::None,
            0u);
    }
    else if constexpr (OP == ApiOp::lgamma_adjusted)
    {
        SFPU_TERNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_lgamma_adjusted, (APPROX, is_fp32_dest_acc_en), t0, t1, t2, t0, VectorMode::RC);
    }
    else if constexpr (OP == ApiOp::mac)
    {
        SFPU_TERNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_mac, (APPROX, is_fp32_dest_acc_en, FMT, 8), t0, t1, t2, t0, VectorMode::RC);
    }
    else
    {
        static_assert(OP == ApiOp::isinf, "api_call: no branch for this op");
    }
}

} // namespace compute_api_perf
} // namespace ckernel
