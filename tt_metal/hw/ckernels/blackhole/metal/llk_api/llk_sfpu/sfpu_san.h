// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"

// SAN_SFPU_TAG(FN, OP) records which SfpuType a ckernel functor implements. The SFPU call macros paste
// FN##_san_tag to recover the op from the functor they are about to invoke, so the identity the sanitizer
// compares is derived from the code that runs rather than from a token written beside it.
//
// The key is the functor NAME, not a function pointer: the op is a property of the function template, and
// every instantiation of calculate_sqrt is sqrt regardless of APPROX, fp32_dest_acc_en or FAST_APPROX.
//
// A functor whose name does not determine the op — the op sits in a template argument, as with
// calculate_sfpu_binary<…, BinaryOp::MUL, …> — is tagged `unused` and stays fenced.
#define SAN_SFPU_TAG(FN, OP)                             \
    struct FN##_san_tag {                                \
        static constexpr ::SfpuType op = ::SfpuType::OP; \
    }

namespace ckernel {

// Which ops the sanitizer models. Modelled ops hook inside their ckernel functor, where the init's
// and the execute's template parameters are both visible; this only decides who keeps the fence.
template <SfpuType OP>
struct sfpu_operation {
    static constexpr bool modelled = false;
};

template <>
struct sfpu_operation<SfpuType::sqrt> {
    static constexpr bool modelled = true;
};

namespace sfpu {

SAN_SFPU_TAG(_add_int_, unused);
SAN_SFPU_TAG(_calculate_ceil_, unused);
SAN_SFPU_TAG(_calculate_comp_unary_int_, unused);
SAN_SFPU_TAG(_calculate_fill_, fill);
SAN_SFPU_TAG(_calculate_fill_bitcast_, fill);
SAN_SFPU_TAG(_calculate_fill_int_, fill);
SAN_SFPU_TAG(_calculate_floor_, unused);
SAN_SFPU_TAG(_calculate_frac_, unused);
SAN_SFPU_TAG(_calculate_negative_, negative);
SAN_SFPU_TAG(_calculate_negative_int_, negative);
SAN_SFPU_TAG(_calculate_round_, unused);
SAN_SFPU_TAG(_calculate_sfpu_binary_bcast_full_tile_, unused);
SAN_SFPU_TAG(_calculate_sfpu_isinf_isnan_, unused);
SAN_SFPU_TAG(_calculate_stochastic_round_, unused);
SAN_SFPU_TAG(_calculate_threshold_, threshold);
SAN_SFPU_TAG(_calculate_trunc_, unused);
SAN_SFPU_TAG(_calculate_where_, where);
SAN_SFPU_TAG(_mul_int_, unused);
SAN_SFPU_TAG(_relu_max_, unused);
SAN_SFPU_TAG(_relu_min_, unused);
SAN_SFPU_TAG(_sub_int_, unused);
SAN_SFPU_TAG(add_int, unused);
SAN_SFPU_TAG(calculate_abs, unused);
SAN_SFPU_TAG(calculate_abs_int32, unused);
SAN_SFPU_TAG(calculate_acos, unused);
SAN_SFPU_TAG(calculate_acosh, unused);
SAN_SFPU_TAG(calculate_activation, unused);
SAN_SFPU_TAG(calculate_add_int32, unused);
SAN_SFPU_TAG(calculate_add_rsqrt, rsqrt);
SAN_SFPU_TAG(calculate_add_top_row, unused);
SAN_SFPU_TAG(calculate_addcdiv, addcdiv);
SAN_SFPU_TAG(calculate_addcmul, addcmul);
SAN_SFPU_TAG(calculate_alt_complex_rotate90, unused);
SAN_SFPU_TAG(calculate_asin, unused);
SAN_SFPU_TAG(calculate_asinh, unused);
SAN_SFPU_TAG(calculate_atan, unused);
SAN_SFPU_TAG(calculate_atanh, unused);
SAN_SFPU_TAG(calculate_binary_comp_fp32, unused);
SAN_SFPU_TAG(calculate_binary_comp_int32, unused);
SAN_SFPU_TAG(calculate_binary_comp_uint, unused);
SAN_SFPU_TAG(calculate_binary_eq_int, unused);
SAN_SFPU_TAG(calculate_binary_left_shift, unused);
SAN_SFPU_TAG(calculate_binary_max_min, unused);
SAN_SFPU_TAG(calculate_binary_max_min_int32, unused);
SAN_SFPU_TAG(calculate_binary_right_shift, unused);
SAN_SFPU_TAG(calculate_binop_with_scalar, unused);
SAN_SFPU_TAG(calculate_bitonic_topk_merge, topk_local_sort);
SAN_SFPU_TAG(calculate_bitonic_topk_phases_steps, topk_local_sort);
SAN_SFPU_TAG(calculate_bitonic_topk_rebuild, topk_local_sort);
SAN_SFPU_TAG(calculate_bitwise_not, bitwise_not);
SAN_SFPU_TAG(calculate_celu, unused);
SAN_SFPU_TAG(calculate_clamp, clamp);
SAN_SFPU_TAG(calculate_clamp_int32, clamp);
SAN_SFPU_TAG(calculate_clamped_logical_right_shift, unused);
SAN_SFPU_TAG(calculate_clamped_silu_glu, unused);
SAN_SFPU_TAG(calculate_comp, unused);
SAN_SFPU_TAG(calculate_comp_int, unused);
SAN_SFPU_TAG(calculate_comp_uint16, unused);
SAN_SFPU_TAG(calculate_comp_unary_int, unused);
SAN_SFPU_TAG(calculate_cosh, unused);
SAN_SFPU_TAG(calculate_cosine, unused);
SAN_SFPU_TAG(calculate_cube_root, cbrt);
SAN_SFPU_TAG(calculate_cumsum, cumsum);
SAN_SFPU_TAG(calculate_dequant_int32, unused);
SAN_SFPU_TAG(calculate_digamma, unused);
SAN_SFPU_TAG(calculate_div_int32, div_int32);
SAN_SFPU_TAG(calculate_div_int32_floor, unused);
SAN_SFPU_TAG(calculate_div_int32_trunc, unused);
SAN_SFPU_TAG(calculate_dropout, dropout);
SAN_SFPU_TAG(calculate_elu, elu);
SAN_SFPU_TAG(calculate_eqz_uint32, unused);
SAN_SFPU_TAG(calculate_erf, unused);
SAN_SFPU_TAG(calculate_erfc, unused);
SAN_SFPU_TAG(calculate_erfinv, erfinv);
SAN_SFPU_TAG(calculate_exp2, unused);
SAN_SFPU_TAG(calculate_expm1, unused);
SAN_SFPU_TAG(calculate_exponential, exponential);
SAN_SFPU_TAG(calculate_fmod, fmod);
SAN_SFPU_TAG(calculate_fmod_int32, unused);
SAN_SFPU_TAG(calculate_fused_max_sub_exp_add_tile, unused);
SAN_SFPU_TAG(calculate_gelu, unused);
SAN_SFPU_TAG(calculate_gelu_derivative_polynomial, unused);
SAN_SFPU_TAG(calculate_gelu_tanh, unused);
SAN_SFPU_TAG(calculate_hardshrink, unused);
SAN_SFPU_TAG(calculate_hardtanh, hardtanh);
SAN_SFPU_TAG(calculate_heaviside, unused);
SAN_SFPU_TAG(calculate_i0, i0);
SAN_SFPU_TAG(calculate_i1, i1);
SAN_SFPU_TAG(calculate_identity, unused);
SAN_SFPU_TAG(calculate_identity_uint, unused);
SAN_SFPU_TAG(calculate_int_mask, mask);
SAN_SFPU_TAG(calculate_left_shift, unused);
SAN_SFPU_TAG(calculate_lerp, lerp);
SAN_SFPU_TAG(calculate_lgamma_adjusted, lgamma);
SAN_SFPU_TAG(calculate_lgamma_stirling, lgamma);
SAN_SFPU_TAG(calculate_lgamma_stirling_fp32, lgamma);
SAN_SFPU_TAG(calculate_log, unused);
SAN_SFPU_TAG(calculate_log1p, log1p);
SAN_SFPU_TAG(calculate_logical_not, logical_not_unary);
SAN_SFPU_TAG(calculate_logical_right_shift, unused);
SAN_SFPU_TAG(calculate_logsigmoid, unused);
SAN_SFPU_TAG(calculate_lrelu, unused);
SAN_SFPU_TAG(calculate_mac, mac);
SAN_SFPU_TAG(calculate_mask, mask);
SAN_SFPU_TAG(calculate_mask_posinf, mask);
SAN_SFPU_TAG(calculate_max_pool_with_indices, unused);
SAN_SFPU_TAG(calculate_mish, mish);
SAN_SFPU_TAG(calculate_nez_uint32, unused);
SAN_SFPU_TAG(calculate_polygamma, polygamma);
SAN_SFPU_TAG(calculate_prelu, prelu);
SAN_SFPU_TAG(calculate_quant_int32, unused);
SAN_SFPU_TAG(calculate_quant_int32_int8_pack, unused);
SAN_SFPU_TAG(calculate_rdiv, rdiv);
SAN_SFPU_TAG(calculate_reciprocal, reciprocal);
SAN_SFPU_TAG(calculate_reduce, unused);
SAN_SFPU_TAG(calculate_remainder, unused);
SAN_SFPU_TAG(calculate_remainder_int32, unused);
SAN_SFPU_TAG(calculate_remainder_uint32, unused);
SAN_SFPU_TAG(calculate_remainder_uint32_scalar, unused);
SAN_SFPU_TAG(calculate_requant_int32, unused);
SAN_SFPU_TAG(calculate_requant_int32_int8_pack, unused);
SAN_SFPU_TAG(calculate_reshuffle_rows, reshuffle_rows);
SAN_SFPU_TAG(calculate_right_shift, unused);
SAN_SFPU_TAG(calculate_rpow, rpow);
SAN_SFPU_TAG(calculate_rsqrt, rsqrt);
SAN_SFPU_TAG(calculate_rsub_int, unused);
SAN_SFPU_TAG(calculate_rsub_scalar_int32, unused);
SAN_SFPU_TAG(calculate_selu, selu);
SAN_SFPU_TAG(calculate_sfpu_atan2, unused);
SAN_SFPU_TAG(calculate_sfpu_binary, unused);
SAN_SFPU_TAG(calculate_sfpu_binary_bitwise, unused);
SAN_SFPU_TAG(calculate_sfpu_binary_div, unused);
SAN_SFPU_TAG(calculate_sfpu_binary_fmod, unused);
SAN_SFPU_TAG(calculate_sfpu_binary_mul, unused);
SAN_SFPU_TAG(calculate_sfpu_binary_pow, unused);
SAN_SFPU_TAG(calculate_sfpu_binary_remainder, unused);
SAN_SFPU_TAG(calculate_sfpu_gcd, gcd);
SAN_SFPU_TAG(calculate_sfpu_isclose, isclose);
SAN_SFPU_TAG(calculate_sfpu_lcm, lcm);
SAN_SFPU_TAG(calculate_sfpu_unary_bitwise, unused);
SAN_SFPU_TAG(calculate_sigmoid, unused);
SAN_SFPU_TAG(calculate_sign, unused);
SAN_SFPU_TAG(calculate_signbit, unused);
SAN_SFPU_TAG(calculate_signbit_int32, unused);
SAN_SFPU_TAG(calculate_silu, unused);
SAN_SFPU_TAG(calculate_sine, unused);
SAN_SFPU_TAG(calculate_sinh, unused);
SAN_SFPU_TAG(calculate_situ_glu, situ_glu);
SAN_SFPU_TAG(calculate_snake_beta, snake_beta);
SAN_SFPU_TAG(calculate_softcap, softcap);
SAN_SFPU_TAG(calculate_softplus, softplus);
SAN_SFPU_TAG(calculate_softshrink, unused);
SAN_SFPU_TAG(calculate_softsign, unused);
SAN_SFPU_TAG(calculate_sqrt, sqrt);
SAN_SFPU_TAG(calculate_square, unused);
SAN_SFPU_TAG(calculate_sub_int32, unused);
SAN_SFPU_TAG(calculate_sum_int_col, unused);
SAN_SFPU_TAG(calculate_sum_int_row, unused);
SAN_SFPU_TAG(calculate_tangent, unused);
SAN_SFPU_TAG(calculate_tanh, unused);
SAN_SFPU_TAG(calculate_tanh_derivative_sech2, tanh_derivative);
SAN_SFPU_TAG(calculate_tanhshrink, unused);
SAN_SFPU_TAG(calculate_tiled_prod, unused);
SAN_SFPU_TAG(calculate_topk_canonicalize_negzero, topk_local_sort);
SAN_SFPU_TAG(calculate_topk_defuse, topk_local_sort);
SAN_SFPU_TAG(calculate_topk_fuse, topk_local_sort);
SAN_SFPU_TAG(calculate_topk_stamp_local_positions, topk_local_sort);
SAN_SFPU_TAG(calculate_topk_stamp_tile_rank_range, topk_local_sort);
SAN_SFPU_TAG(calculate_typecast, typecast);
SAN_SFPU_TAG(calculate_typecast_fp32_to_fp16b, typecast);
SAN_SFPU_TAG(calculate_typecast_fp32_to_int32, typecast);
SAN_SFPU_TAG(calculate_typecast_fp32_to_uint16, typecast);
SAN_SFPU_TAG(calculate_typecast_fp32_to_uint32, typecast);
SAN_SFPU_TAG(calculate_typecast_fp32_to_uint8, typecast);
SAN_SFPU_TAG(calculate_typecast_int32_to_fp16b, typecast);
SAN_SFPU_TAG(calculate_typecast_int32_to_fp32, typecast);
SAN_SFPU_TAG(calculate_typecast_int32_to_uint16, typecast);
SAN_SFPU_TAG(calculate_typecast_int8_to_fp32, typecast);
SAN_SFPU_TAG(calculate_typecast_int8_to_int32, typecast);
SAN_SFPU_TAG(calculate_typecast_uint16_to_fp16b, typecast);
SAN_SFPU_TAG(calculate_typecast_uint16_to_fp32, typecast);
SAN_SFPU_TAG(calculate_typecast_uint16_to_uint32, typecast);
SAN_SFPU_TAG(calculate_typecast_uint32_to_fp16b, typecast);
SAN_SFPU_TAG(calculate_typecast_uint32_to_fp32, typecast);
SAN_SFPU_TAG(calculate_typecast_uint32_to_uint16, typecast);
SAN_SFPU_TAG(calculate_typecast_uint_to_uint8, typecast);
SAN_SFPU_TAG(calculate_unary_eq, unused);
SAN_SFPU_TAG(calculate_unary_ge, unused);
SAN_SFPU_TAG(calculate_unary_gt, unused);
SAN_SFPU_TAG(calculate_unary_le, unused);
SAN_SFPU_TAG(calculate_unary_lt, unused);
SAN_SFPU_TAG(calculate_unary_max_min, unused);
SAN_SFPU_TAG(calculate_unary_max_min_int32, unused);
SAN_SFPU_TAG(calculate_unary_ne, unused);
SAN_SFPU_TAG(calculate_unary_power, unused);
SAN_SFPU_TAG(calculate_unary_power_iterative, unused);
SAN_SFPU_TAG(calculate_xielu, xielu);
SAN_SFPU_TAG(calculate_zero_comp, unused);
SAN_SFPU_TAG(copy_dest_value, unused);
SAN_SFPU_TAG(generalized_moe_gate_copy_topk_run, unused);
SAN_SFPU_TAG(generalized_moe_gate_finalize_ungrouped, unused);
SAN_SFPU_TAG(generalized_moe_gate_merge16_to_run, unused);
SAN_SFPU_TAG(generalized_moe_gate_merge4_top8, unused);
SAN_SFPU_TAG(generalized_moe_gate_place_field_from_interm, unused);
SAN_SFPU_TAG(generalized_moe_gate_sort_top4_groups, unused);
SAN_SFPU_TAG(generalized_moe_gate_sum_top2, unused);
SAN_SFPU_TAG(generalized_moe_gate_top8, unused);
SAN_SFPU_TAG(hardmish, hardmish);
SAN_SFPU_TAG(mul_int32, unused);
SAN_SFPU_TAG(rand, unused);
SAN_SFPU_TAG(relu_clamp_int, unused);
SAN_SFPU_TAG(relu_clamp_uint, unused);
}  // namespace sfpu
}  // namespace ckernel
