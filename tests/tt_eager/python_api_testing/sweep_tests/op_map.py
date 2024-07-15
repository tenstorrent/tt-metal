# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)


op_map = {
    ################################################
    ################# Helper-Funcs #################
    ################################################
    "linear": {"tt_op": tt_lib_ops.linear, "pytorch_op": pytorch_ops.linear},
    ################################################
    #################### TT-LIB ####################
    ################################################
    "clone": {
        "tt_op": tt_lib_ops.clone,
        "pytorch_op": pytorch_ops.clone,
    },
    "typecast": {
        "tt_op": tt_lib_ops.typecast,
        "pytorch_op": pytorch_ops.typecast,
    },
    "copy": {
        "tt_op": tt_lib_ops.copy,
        "pytorch_op": pytorch_ops.copy,
    },
    "concat": {
        "tt_op": tt_lib_ops.concat,
        "pytorch_op": pytorch_ops.concat,
    },
    "move": {
        "tt_op": tt_lib_ops.move,
        "pytorch_op": pytorch_ops.move,
    },
    "arange": {
        "tt_op": tt_lib_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    "prod": {
        "tt_op": tt_lib_ops.prod,
        "pytorch_op": pytorch_ops.prod,
    },
    # stats
    "stats-var_hw": {
        "tt_op": tt_lib_ops.var_hw,
        "pytorch_op": pytorch_ops.var_hw,
    },
    "stats-std_hw": {
        "tt_op": tt_lib_ops.std_hw,
        "pytorch_op": pytorch_ops.std_hw,
    },
    "stats-mean_hw": {
        "tt_op": tt_lib_ops.mean_hw,
        "pytorch_op": pytorch_ops.mean_hw,
    },
    "stats-normalize_hw": {
        "tt_op": tt_lib_ops.normalize_hw,
        "pytorch_op": pytorch_ops.normalize_hw,
    },
    "stats-var_global": {
        "tt_op": None,  # tt_lib_ops.var_global,
        "pytorch_op": pytorch_ops.var_global,
    },
    "stats-std_global": {
        "tt_op": None,  # tt_lib_ops.std_global,
        "pytorch_op": pytorch_ops.std_global,
    },
    "stats-mean_global": {
        "tt_op": None,  # tt_lib_ops.mean_global,
        "pytorch_op": pytorch_ops.mean_global,
    },
    "stats-normalize_global": {
        "tt_op": tt_lib_ops.normalize_global,
        "pytorch_op": pytorch_ops.normalize_global,
    },
    # Eltwise unary
    "eltwise-hardtanh": {
        "tt_op": tt_lib_ops.eltwise_hardtanh,
        "pytorch_op": pytorch_ops.hardtanh,
    },
    "eltwise-clip": {
        "tt_op": tt_lib_ops.clip,
        "pytorch_op": pytorch_ops.clip,
    },
    "eltwise-tril": {
        "tt_op": tt_lib_ops.tril,
        "pytorch_op": pytorch_ops.tril,
    },
    "eltwise-triu": {
        "tt_op": tt_lib_ops.triu,
        "pytorch_op": pytorch_ops.triu,
    },
    "eltwise-zeros": {
        "tt_op": tt_lib_ops.zeros,
        "pytorch_op": pytorch_ops.zeros,
    },
    "eltwise-empty": {
        "tt_op": tt_lib_ops.empty,
        "pytorch_op": pytorch_ops.empty,
    },
    "eltwise-ones": {
        "tt_op": tt_lib_ops.ones,
        "pytorch_op": pytorch_ops.ones,
    },
    "fill-rm": {
        "tt_op": tt_lib_ops.fill_rm,
        "pytorch_op": pytorch_ops.fill_rm,
    },
    "fill-ones-rm": {
        "tt_op": tt_lib_ops.fill_ones_rm,
        "pytorch_op": pytorch_ops.fill_ones_rm,
    },
    "fill-bw": {
        "tt_op": tt_lib_ops.fill_bw,
        "pytorch_op": pytorch_ops.fill_bw,
    },
    "fill-zero-bw": {
        "tt_op": tt_lib_ops.fill_zero_bw,
        "pytorch_op": pytorch_ops.fill_zero_bw,
    },
    "eltwise-full": {
        "tt_op": tt_lib_ops.full,
        "pytorch_op": pytorch_ops.full,
    },
    "eltwise-zeros_like": {
        "tt_op": tt_lib_ops.zeros_like,
        "pytorch_op": pytorch_ops.zeros_like,
    },
    "eltwise-ones_like": {
        "tt_op": tt_lib_ops.ones_like,
        "pytorch_op": pytorch_ops.ones_like,
    },
    "eltwise-full_like": {
        "tt_op": tt_lib_ops.full_like,
        "pytorch_op": pytorch_ops.full_like,
    },
    "eltwise-div_unary": {
        "tt_op": tt_lib_ops.eltwise_div_unary,
        "pytorch_op": pytorch_ops.div_unary,
    },
    "eltwise-unary_div": {
        "tt_op": tt_lib_ops.eltwise_unary_div,
        "pytorch_op": pytorch_ops.unary_div,
    },
    "eltwise-mul_unary": {
        "tt_op": tt_lib_ops.eltwise_mul_unary,
        "pytorch_op": pytorch_ops.mul_unary,
    },
    "eltwise-sub_unary": {
        "tt_op": tt_lib_ops.eltwise_sub_unary,
        "pytorch_op": pytorch_ops.sub_unary,
    },
    "sub-unary-bw": {
        "tt_op": tt_lib_ops.sub_unary_bw,
        "pytorch_op": pytorch_ops.sub_unary_bw,
    },
    "eltwise-add_unary": {
        "tt_op": tt_lib_ops.eltwise_add_unary,
        "pytorch_op": pytorch_ops.add_unary,
    },
    "eltwise-logical_not_unary": {
        "tt_op": tt_lib_ops.eltwise_logical_not_unary,
        "pytorch_op": pytorch_ops.logical_not_unary,
    },
    "eltwise-i0": {
        "tt_op": tt_lib_ops.eltwise_i0,
        "pytorch_op": pytorch_ops.i0,
    },
    "eltwise-lgamma": {
        "tt_op": tt_lib_ops.eltwise_lgamma,
        "pytorch_op": pytorch_ops.lgamma,
    },
    "eltwise-logical_noti": {
        "tt_op": tt_lib_ops.eltwise_logical_noti,
        "pytorch_op": pytorch_ops.logical_noti,
    },
    "eltwise-bitwise_complement": {
        "tt_op": None,  # tt_lib_ops.eltwise_bitwise_complement,
        "pytorch_op": None,  # pytorch_ops.bitwise_complement,
    },
    "eltwise-logical_xor": {
        "tt_op": tt_lib_ops.eltwise_logical_xor,
        "pytorch_op": pytorch_ops.logical_xor,
    },
    "eltwise-sinh": {
        "tt_op": tt_lib_ops.eltwise_sinh,
        "pytorch_op": pytorch_ops.sinh,
    },
    "eltwise-cosh": {
        "tt_op": tt_lib_ops.eltwise_cosh,
        "pytorch_op": pytorch_ops.cosh,
    },
    "eltwise-ltz": {
        "tt_op": tt_lib_ops.eltwise_ltz,
        "pytorch_op": pytorch_ops.ltz,
    },
    "eltwise-gtz": {
        "tt_op": tt_lib_ops.eltwise_gtz,
        "pytorch_op": pytorch_ops.gtz,
    },
    "eltwise-lez": {
        "tt_op": tt_lib_ops.eltwise_lez,
        "pytorch_op": pytorch_ops.lez,
    },
    "eltwise-gez": {
        "tt_op": tt_lib_ops.eltwise_gez,
        "pytorch_op": pytorch_ops.gez,
    },
    "eltwise-eqz": {
        "tt_op": tt_lib_ops.eltwise_eqz,
        "pytorch_op": pytorch_ops.eqz,
    },
    "eltwise-nez": {
        "tt_op": tt_lib_ops.eltwise_nez,
        "pytorch_op": pytorch_ops.nez,
    },
    "eltwise-abs": {
        "tt_op": tt_lib_ops.eltwise_abs,
        "pytorch_op": pytorch_ops.abs,
    },
    "eltwise-digamma": {
        "tt_op": tt_lib_ops.eltwise_digamma,
        "pytorch_op": pytorch_ops.digamma,
    },
    "eltwise-isfinite": {
        "tt_op": tt_lib_ops.eltwise_isfinite,
        "pytorch_op": pytorch_ops.isfinite,
    },
    "eltwise-isinf": {
        "tt_op": tt_lib_ops.eltwise_isinf,
        "pytorch_op": pytorch_ops.isinf,
    },
    "eltwise-isposinf": {
        "tt_op": tt_lib_ops.eltwise_isposinf,
        "pytorch_op": pytorch_ops.isposinf,
    },
    "eltwise-isneginf": {
        "tt_op": tt_lib_ops.eltwise_isneginf,
        "pytorch_op": pytorch_ops.isneginf,
    },
    "eltwise-isnan": {
        "tt_op": tt_lib_ops.eltwise_isnan,
        "pytorch_op": pytorch_ops.isnan,
    },
    "eltwise-sign": {
        "tt_op": tt_lib_ops.eltwise_sign,
        "pytorch_op": pytorch_ops.sign,
    },
    "eltwise-multigammaln": {
        "tt_op": tt_lib_ops.eltwise_multigammaln,
        "pytorch_op": pytorch_ops.multigammaln,
    },
    "eltwise-silu": {
        "tt_op": tt_lib_ops.eltwise_silu,
        "pytorch_op": pytorch_ops.silu,
    },
    "eltwise-elu": {
        "tt_op": tt_lib_ops.eltwise_elu,
        "pytorch_op": pytorch_ops.elu,
    },
    "eltwise-div": {
        "tt_op": tt_lib_ops.eltwise_div,
        "pytorch_op": pytorch_ops.div,
    },
    "eltwise-div_trunc": {
        "tt_op": tt_lib_ops.eltwise_div_trunc,
        "pytorch_op": pytorch_ops.div_trunc,
    },
    "eltwise-unary_div_trunc": {
        "tt_op": tt_lib_ops.eltwise_unary_div_trunc,
        "pytorch_op": pytorch_ops.unary_div_trunc,
    },
    "eltwise-unary_rdiv_trunc": {
        "tt_op": tt_lib_ops.eltwise_unary_rdiv_trunc,
        "pytorch_op": pytorch_ops.unary_rdiv_trunc,
    },
    "eltwise-div_no_nan": {
        "tt_op": tt_lib_ops.eltwise_div_no_nan,
        "pytorch_op": pytorch_ops.div_no_nan,
    },
    "eltwise-unary_div_no_nan": {
        "tt_op": tt_lib_ops.eltwise_unary_div_no_nan,
        "pytorch_op": pytorch_ops.unary_div_no_nan,
    },
    "eltwise-square": {
        "tt_op": tt_lib_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "eltwise-mish": {
        "tt_op": tt_lib_ops.eltwise_mish,
        "pytorch_op": pytorch_ops.mish,
    },
    "eltwise-softplus": {
        "tt_op": tt_lib_ops.eltwise_softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "eltwise-log1p": {
        "tt_op": tt_lib_ops.eltwise_log1p,
        "pytorch_op": pytorch_ops.log1p,
    },
    "eltwise-neg": {
        "tt_op": tt_lib_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
    },
    "eltwise-swish": {
        "tt_op": tt_lib_ops.eltwise_swish,
        "pytorch_op": pytorch_ops.swish,
    },
    "eltwise-cos": {
        "tt_op": tt_lib_ops.eltwise_cos,
        "pytorch_op": pytorch_ops.cos,
    },
    "eltwise-sin": {
        "tt_op": tt_lib_ops.eltwise_sin,
        "pytorch_op": pytorch_ops.sin,
    },
    "eltwise-tan": {
        "tt_op": tt_lib_ops.eltwise_tan,
        "pytorch_op": pytorch_ops.tan,
    },
    "eltwise-tan-bw": {
        "tt_op": tt_lib_ops.eltwise_tan_bw,
        "pytorch_op": pytorch_ops.tan_bw,
    },
    "eltwise-asin": {
        "tt_op": tt_lib_ops.eltwise_asin,
        "pytorch_op": pytorch_ops.asin,
    },
    "eltwise-atan": {
        "tt_op": tt_lib_ops.eltwise_atan,
        "pytorch_op": pytorch_ops.atan,
    },
    "eltwise-atanh": {
        "tt_op": tt_lib_ops.eltwise_atanh,
        "pytorch_op": pytorch_ops.atanh,
    },
    "eltwise-acos": {
        "tt_op": tt_lib_ops.eltwise_acos,
        "pytorch_op": pytorch_ops.acos,
    },
    "eltwise-asinh": {
        "tt_op": tt_lib_ops.eltwise_asinh,
        "pytorch_op": pytorch_ops.asinh,
    },
    "eltwise-acosh": {
        "tt_op": tt_lib_ops.eltwise_acosh,
        "pytorch_op": pytorch_ops.acosh,
    },
    "eltwise-exp": {
        "tt_op": tt_lib_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-exp-bw": {
        "tt_op": tt_lib_ops.eltwise_exp_bw,
        "pytorch_op": pytorch_ops.exp_bw,
    },
    "eltwise-exp2": {
        "tt_op": tt_lib_ops.eltwise_exp2,
        "pytorch_op": pytorch_ops.exp2,
    },
    "eltwise-expm1": {
        "tt_op": tt_lib_ops.eltwise_expm1,
        "pytorch_op": pytorch_ops.expm1,
    },
    "eltwise-recip": {
        "tt_op": tt_lib_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "tt_op": tt_lib_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "tt_op": tt_lib_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-softmax_in_place": {
        "tt_op": tt_lib_ops.eltwise_softmax_in_place,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "eltwise-scale_mask_softmax_in_place": {
        "tt_op": tt_lib_ops.eltwise_scale_mask_softmax_in_place,
        "pytorch_op": pytorch_ops.scale_mask_softmax_in_place,
    },
    "eltwise-rsqrt": {
        "tt_op": tt_lib_ops.eltwise_rsqrt,
        "pytorch_op": pytorch_ops.rsqrt,
    },
    "eltwise-xlogy": {
        "tt_op": tt_lib_ops.eltwise_xlogy,
        "pytorch_op": pytorch_ops.xlogy,
    },
    "eltwise-logical_and": {
        "tt_op": tt_lib_ops.eltwise_logical_and,
        "pytorch_op": pytorch_ops.logical_and,
    },
    "eltwise-logical_andi": {
        "tt_op": tt_lib_ops.eltwise_logical_andi,
        "pytorch_op": pytorch_ops.logical_andi,
    },
    "eltwise-atan2": {
        "tt_op": tt_lib_ops.eltwise_atan2,
        "pytorch_op": pytorch_ops.atan2,
    },
    "eltwise-lerp_binary": {
        "tt_op": tt_lib_ops.eltwise_lerp_binary,
        "pytorch_op": pytorch_ops.lerp_binary,
    },
    "eltwise-lerp_ternary": {
        "tt_op": tt_lib_ops.eltwise_lerp_ternary,
        "pytorch_op": pytorch_ops.lerp_ternary,
    },
    "eltwise-leaky_relu": {
        "tt_op": tt_lib_ops.eltwise_leaky_relu,
        "pytorch_op": pytorch_ops.leaky_relu,
    },
    "eltwise-prelu": {
        "tt_op": tt_lib_ops.eltwise_prelu,
        "pytorch_op": pytorch_ops.prelu,
    },
    "eltwise-hardshrink": {
        "tt_op": tt_lib_ops.eltwise_hardshrink,
        "pytorch_op": pytorch_ops.hardshrink,
    },
    "eltwise-bias_gelu_unary": {
        "tt_op": tt_lib_ops.eltwise_bias_gelu_unary,
        "pytorch_op": pytorch_ops.bias_gelu_unary,
    },
    "eltwise-softshrink": {
        "tt_op": tt_lib_ops.eltwise_softshrink,
        "pytorch_op": pytorch_ops.softshrink,
    },
    "eltwise-softsign": {
        "tt_op": tt_lib_ops.eltwise_softsign,
        "pytorch_op": pytorch_ops.softsign,
    },
    "eltwise-relu": {
        "tt_op": tt_lib_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "eltwise-pow": {
        "tt_op": tt_lib_ops.eltwise_pow,
        "pytorch_op": pytorch_ops.power,
    },
    "unary-pow-bw": {
        "tt_op": tt_lib_ops.unary_pow_bw,
        "pytorch_op": pytorch_ops.power_bw,
    },
    "bert-large-fused-qkv-matmul": {
        "tt_op": tt_lib_ops.bert_large_fused_qkv_matmul,
        "pytorch_op": pytorch_ops.bert_large_fused_qkv_matmul,
    },
    "eltwise-relu_min": {
        "tt_op": tt_lib_ops.eltwise_relu_min,
        "pytorch_op": pytorch_ops.relu_min,
    },
    "eltwise-polyval": {
        "tt_op": tt_lib_ops.eltwise_polyval,
        "pytorch_op": pytorch_ops.polyval,
    },
    "eltwise-mac": {
        "tt_op": tt_lib_ops.eltwise_mac,
        "pytorch_op": pytorch_ops.mac,
    },
    "eltwise-addcmul": {
        "tt_op": tt_lib_ops.eltwise_addcmul,
        "pytorch_op": pytorch_ops.addcmul,
    },
    "eltwise-celu": {
        "tt_op": tt_lib_ops.eltwise_celu,
        "pytorch_op": pytorch_ops.celu,
    },
    "eltwise-addcdiv": {
        "tt_op": tt_lib_ops.eltwise_addcdiv,
        "pytorch_op": pytorch_ops.addcdiv,
    },
    "eltwise-sigmoid": {
        "tt_op": tt_lib_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-sigmoid_accurate": {
        "tt_op": tt_lib_ops.eltwise_sigmoid_accurate,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-log_sigmoid": {
        "tt_op": tt_lib_ops.eltwise_log_sigmoid,
        "pytorch_op": pytorch_ops.log_sigmoid,
    },
    "eltwise-heaviside": {
        "tt_op": tt_lib_ops.eltwise_heaviside,
        "pytorch_op": pytorch_ops.heaviside,
    },
    "eltwise-bitwise_xor": {
        "tt_op": tt_lib_ops.eltwise_bitwise_xor,
        "pytorch_op": pytorch_ops.bitwise_xor,
    },
    "eltwise-bitwise_not": {
        "tt_op": tt_lib_ops.eltwise_bitwise_not,
        "pytorch_op": pytorch_ops.bitwise_not,
    },
    "eltwise-bitwise_and": {
        "tt_op": tt_lib_ops.eltwise_bitwise_and,
        "pytorch_op": pytorch_ops.bitwise_and,
    },
    "eltwise-bitwise_or": {
        "tt_op": tt_lib_ops.eltwise_bitwise_or,
        "pytorch_op": pytorch_ops.bitwise_or,
    },
    "eltwise-right_shift": {
        "tt_op": tt_lib_ops.eltwise_right_shift,
        "pytorch_op": pytorch_ops.right_shift,
    },
    "eltwise-left_shift": {
        "tt_op": tt_lib_ops.eltwise_left_shift,
        "pytorch_op": pytorch_ops.left_shift,
    },
    "eltwise-unary_remainder": {
        "tt_op": tt_lib_ops.eltwise_unary_remainder,
        "pytorch_op": pytorch_ops.unary_remainder,
    },
    "eltwise-remainder": {
        "tt_op": tt_lib_ops.eltwise_remainder,
        "pytorch_op": pytorch_ops.remainder,
    },
    "eltwise-fmod": {
        "tt_op": tt_lib_ops.eltwise_fmod,
        "pytorch_op": pytorch_ops.fmod,
    },
    "eltwise-unary_fmod": {
        "tt_op": tt_lib_ops.eltwise_unary_fmod,
        "pytorch_op": pytorch_ops.unary_fmod,
    },
    "eltwise-unary_ne": {
        "tt_op": tt_lib_ops.eltwise_unary_ne,
        "pytorch_op": pytorch_ops.unary_ne,
    },
    "eltwise-erf": {
        "tt_op": tt_lib_ops.eltwise_erf,
        "pytorch_op": pytorch_ops.erf,
    },
    "eltwise-erfc": {
        "tt_op": tt_lib_ops.eltwise_erfc,
        "pytorch_op": pytorch_ops.erfc,
    },
    "eltwise-erfinv": {
        "tt_op": tt_lib_ops.eltwise_erfinv,
        "pytorch_op": pytorch_ops.erfinv,
    },
    "eltwise-nextafter": {
        "tt_op": tt_lib_ops.eltwise_nextafter,
        "pytorch_op": pytorch_ops.nextafter,
    },
    "eltwise-subalpha": {
        "tt_op": tt_lib_ops.eltwise_subalpha,
        "pytorch_op": pytorch_ops.subalpha,
    },
    "eltwise-addalpha": {
        "tt_op": tt_lib_ops.eltwise_addalpha,
        "pytorch_op": pytorch_ops.addalpha,
    },
    "eltwise-addalpha-optional": {
        "tt_op": tt_lib_ops.eltwise_addalpha_optional,
        "pytorch_op": pytorch_ops.addalpha,
    },
    "lamb-optimizer": {
        "tt_op": tt_lib_ops.lamb_optimizer,
        "pytorch_op": pytorch_ops.lamb_optimizer,
    },
    "eltwise-logit": {
        "tt_op": tt_lib_ops.eltwise_logit,
        "pytorch_op": pytorch_ops.logit,
    },
    "eltwise-polygamma": {
        "tt_op": tt_lib_ops.eltwise_polygamma,
        "pytorch_op": pytorch_ops.polygamma,
    },
    "eltwise-logical_xori": {
        "tt_op": tt_lib_ops.eltwise_logical_xori,
        "pytorch_op": pytorch_ops.logical_xori,
    },
    "eltwise-hardsigmoid": {
        "tt_op": tt_lib_ops.eltwise_hardsigmoid,
        "pytorch_op": pytorch_ops.hardsigmoid,
    },
    "eltwise-hardswish": {
        "tt_op": tt_lib_ops.eltwise_hardswish,
        "pytorch_op": pytorch_ops.hardswish,
    },
    "eltwise-log": {
        "tt_op": tt_lib_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "eltwise-log2": {
        "tt_op": tt_lib_ops.eltwise_log2,
        "pytorch_op": pytorch_ops.log2,
    },
    "eltwise-log10": {
        "tt_op": tt_lib_ops.eltwise_log10,
        "pytorch_op": pytorch_ops.log10,
    },
    "eltwise-tanh": {
        "tt_op": tt_lib_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    "eltwise-tanh-bw": {
        "tt_op": tt_lib_ops.eltwise_tanh_bw,
        "pytorch_op": pytorch_ops.tanh_bw,
    },
    "eltwise-tanhshrink": {
        "tt_op": tt_lib_ops.eltwise_tanhshrink,
        "pytorch_op": pytorch_ops.tanhshrink,
    },
    "eltwise-signbit": {
        "tt_op": tt_lib_ops.eltwise_signbit,
        "pytorch_op": pytorch_ops.signbit,
    },
    "eltwise-floor": {
        "tt_op": tt_lib_ops.eltwise_floor,
        "pytorch_op": pytorch_ops.floor,
    },
    "eltwise-ceil": {
        "tt_op": tt_lib_ops.eltwise_ceil,
        "pytorch_op": pytorch_ops.ceil,
    },
    "eltwise-trunc": {
        "tt_op": tt_lib_ops.eltwise_trunc,
        "pytorch_op": pytorch_ops.trunc,
    },
    "eltwise-frac": {
        "tt_op": tt_lib_ops.eltwise_frac,
        "pytorch_op": pytorch_ops.frac,
    },
    "eltwise-floor_div": {
        "tt_op": tt_lib_ops.eltwise_floor_div,
        "pytorch_op": pytorch_ops.floor_div,
    },
    "eltwise-unary_floor_div": {
        "tt_op": tt_lib_ops.eltwise_unary_floor_div,
        "pytorch_op": pytorch_ops.unary_floor_div,
    },
    "eltwise-rfloor_div": {
        "tt_op": tt_lib_ops.eltwise_rfloor_div,
        "pytorch_op": pytorch_ops.rfloor_div,
    },
    "eltwise-round": {
        "tt_op": tt_lib_ops.eltwise_round,
        "pytorch_op": pytorch_ops.round,
    },
    "eltwise-rpow": {
        "tt_op": tt_lib_ops.eltwise_rpow,
        "pytorch_op": pytorch_ops.eltwise_rpow,
    },
    "eltwise-rdiv": {
        "tt_op": tt_lib_ops.eltwise_rdiv,
        "pytorch_op": pytorch_ops.eltwise_rdiv,
    },
    "eltwise-rsub": {
        "tt_op": tt_lib_ops.eltwise_rsub,
        "pytorch_op": pytorch_ops.eltwise_rsub,
    },
    "eltwise-identity": {
        "tt_op": tt_lib_ops.eltwise_identity,
        "pytorch_op": pytorch_ops.eltwise_identity,
    },
    "eltwise-typecast": {
        "tt_op": tt_lib_ops.eltwise_typecast,
        "pytorch_op": pytorch_ops.eltwise_typecast,
    },
    "eltwise-unary_gt": {
        "tt_op": tt_lib_ops.eltwise_unary_gt,
        "pytorch_op": pytorch_ops.unary_gt,
    },
    "eltwise-unary_lt": {
        "tt_op": tt_lib_ops.eltwise_unary_lt,
        "pytorch_op": pytorch_ops.unary_lt,
    },
    # Eltwise binary
    "eltwise-ne": {
        "tt_op": tt_lib_ops.eltwise_ne,
        "pytorch_op": pytorch_ops.ne,
    },
    "eltwise-bias_gelu": {
        "tt_op": tt_lib_ops.eltwise_bias_gelu,
        "pytorch_op": pytorch_ops.bias_gelu,
    },
    "eltwise-eq": {
        "tt_op": tt_lib_ops.eltwise_eq,
        "pytorch_op": pytorch_ops.eq,
    },
    "eltwise-lt": {
        "tt_op": tt_lib_ops.eltwise_lt,
        "pytorch_op": pytorch_ops.lt,
    },
    "eltwise-gt": {
        "tt_op": tt_lib_ops.eltwise_gt,
        "pytorch_op": pytorch_ops.gt,
    },
    "eltwise-gte": {
        "tt_op": tt_lib_ops.eltwise_gte,
        "pytorch_op": pytorch_ops.gte,
    },
    "eltwise-lte": {
        "tt_op": tt_lib_ops.eltwise_lte,
        "pytorch_op": pytorch_ops.lte,
    },
    "eltwise-add": {
        "tt_op": tt_lib_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-add-bw": {
        "tt_op": tt_lib_ops.eltwise_add_bw,
        "pytorch_op": pytorch_ops.add_bw,
    },
    "eltwise-sub": {
        "tt_op": tt_lib_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-sub-bw": {
        "tt_op": tt_lib_ops.eltwise_sub_bw,
        "pytorch_op": pytorch_ops.sub_bw,
    },
    "eltwise-mul": {
        "tt_op": tt_lib_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    "eltwise-mul-bw": {
        "tt_op": tt_lib_ops.eltwise_mul_bw,
        "pytorch_op": pytorch_ops.mul_bw,
    },
    "eltwise-min": {
        "tt_op": tt_lib_ops.eltwise_min,
        "pytorch_op": pytorch_ops.min,
    },
    "eltwise-min-bw": {
        "tt_op": tt_lib_ops.eltwise_min_bw,
        "pytorch_op": pytorch_ops.min_bw,
    },
    "eltwise-max": {
        "tt_op": tt_lib_ops.eltwise_max,
        "pytorch_op": pytorch_ops.max,
    },
    "eltwise-max-bw": {
        "tt_op": tt_lib_ops.eltwise_max_bw,
        "pytorch_op": pytorch_ops.max_bw,
    },
    "eltwise-squared_difference": {
        "tt_op": tt_lib_ops.eltwise_squared_difference,
        "pytorch_op": pytorch_ops.squared_difference,
    },
    "eltwise-deg2rad": {
        "tt_op": tt_lib_ops.eltwise_deg2rad,
        "pytorch_op": pytorch_ops.deg2rad,
    },
    "eltwise-rad2deg": {
        "tt_op": tt_lib_ops.eltwise_rad2deg,
        "pytorch_op": pytorch_ops.rad2deg,
    },
    "eltwise-cbrt": {
        "tt_op": tt_lib_ops.eltwise_cbrt,
        "pytorch_op": pytorch_ops.cbrt,
    },
    "eltwise-hypot": {
        "tt_op": tt_lib_ops.eltwise_hypot,
        "pytorch_op": pytorch_ops.hypot,
    },
    "eltwise-scatter": {
        "tt_op": tt_lib_ops.eltwise_scatter,
        "pytorch_op": pytorch_ops.scatter,
    },
    "eltwise-threshold": {
        "tt_op": tt_lib_ops.eltwise_threshold,
        "pytorch_op": pytorch_ops.threshold,
    },
    "eltwise-relu6": {
        "tt_op": tt_lib_ops.eltwise_relu6,
        "pytorch_op": pytorch_ops.relu6,
    },
    "eltwise-ldexp": {
        "tt_op": tt_lib_ops.eltwise_ldexp,
        "pytorch_op": pytorch_ops.ldexp,
    },
    "eltwise-logaddexp": {
        "tt_op": tt_lib_ops.eltwise_logaddexp,
        "pytorch_op": pytorch_ops.logaddexp,
    },
    "eltwise-logaddexp2": {
        "tt_op": tt_lib_ops.eltwise_logaddexp2,
        "pytorch_op": pytorch_ops.logaddexp2,
    },
    "eltwise-assign_binary": {
        "tt_op": tt_lib_ops.eltwise_assign_binary,
        "pytorch_op": pytorch_ops.assign_binary,
    },
    "eltwise-assign_unary": {
        "tt_op": tt_lib_ops.eltwise_assign_unary,
        "pytorch_op": pytorch_ops.assign_unary,
    },
    "eltwise-logical_or": {
        "tt_op": tt_lib_ops.eltwise_logical_or,
        "pytorch_op": pytorch_ops.logical_or,
    },
    "eltwise-logical_ori": {
        "tt_op": tt_lib_ops.eltwise_logical_ori,
        "pytorch_op": pytorch_ops.logical_ori,
    },
    "eltwise-isclose": {
        "tt_op": tt_lib_ops.eltwise_isclose,
        "pytorch_op": pytorch_ops.isclose,
    },
    # Eltwise binary with optional output
    "eltwise-ne-optional": {
        "tt_op": tt_lib_ops.eltwise_ne_optional,
        "pytorch_op": pytorch_ops.ne,
    },
    "eltwise-bias_gelu-optional": {
        "tt_op": tt_lib_ops.eltwise_bias_gelu_optional,
        "pytorch_op": pytorch_ops.bias_gelu,
    },
    "eltwise-eq-optional": {
        "tt_op": tt_lib_ops.eltwise_eq_optional,
        "pytorch_op": pytorch_ops.eq,
    },
    "eltwise-lt-optional": {
        "tt_op": tt_lib_ops.eltwise_lt_optional,
        "pytorch_op": pytorch_ops.lt,
    },
    "eltwise-gt-optional": {
        "tt_op": tt_lib_ops.eltwise_gt_optional,
        "pytorch_op": pytorch_ops.gt,
    },
    "eltwise-gte-optional": {
        "tt_op": tt_lib_ops.eltwise_gte_optional,
        "pytorch_op": pytorch_ops.gte,
    },
    "eltwise-lte-optional": {
        "tt_op": tt_lib_ops.eltwise_lte_optional,
        "pytorch_op": pytorch_ops.lte,
    },
    "eltwise-add-optional": {
        "tt_op": tt_lib_ops.eltwise_add_optional,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-sub-optional": {
        "tt_op": tt_lib_ops.eltwise_sub_optional,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-mul-optional": {
        "tt_op": tt_lib_ops.eltwise_mul_optional,
        "pytorch_op": pytorch_ops.mul,
    },
    "eltwise-squared_difference-optional": {
        "tt_op": tt_lib_ops.eltwise_squared_difference_optional,
        "pytorch_op": pytorch_ops.squared_difference,
    },
    "eltwise-ldexp-optional": {
        "tt_op": tt_lib_ops.eltwise_ldexp_optional,
        "pytorch_op": pytorch_ops.ldexp,
    },
    "eltwise-logaddexp-optional": {
        "tt_op": tt_lib_ops.eltwise_logaddexp_optional,
        "pytorch_op": pytorch_ops.logaddexp,
    },
    "eltwise-logaddexp2-optional": {
        "tt_op": tt_lib_ops.eltwise_logaddexp2_optional,
        "pytorch_op": pytorch_ops.logaddexp2,
    },
    "eltwise-logical_or-optional": {
        "tt_op": tt_lib_ops.eltwise_logical_or_optional,
        "pytorch_op": pytorch_ops.logical_or,
    },
    "eltwise-logical_and-optional": {
        "tt_op": tt_lib_ops.eltwise_logical_and_optional,
        "pytorch_op": pytorch_ops.logical_and,
    },
    # Eltwise ternary
    "eltwise-arange": {
        "tt_op": tt_lib_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    "eltwise-where": {
        "tt_op": tt_lib_ops.where,
        "pytorch_op": pytorch_ops.where,
    },
    "eltwise-where-optional": {
        "tt_op": tt_lib_ops.where_optional,
        "pytorch_op": pytorch_ops.where,
    },
    "eltwise-where-scalar-optional": {
        "tt_op": tt_lib_ops.where_scalar_optional,
        "pytorch_op": pytorch_ops.where_scalar,
    },
    "where-bw": {
        "tt_op": tt_lib_ops.where_bw,
        "pytorch_op": pytorch_ops.where_bw,
    },
    # Matmul
    "matmul": {
        "tt_op": tt_lib_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "outer": {
        "tt_op": tt_lib_ops.outer,
        "pytorch_op": pytorch_ops.outer,
    },
    # Broadcast
    "bcast-add-h": {
        "tt_op": tt_lib_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "tt_op": tt_lib_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "tt_op": tt_lib_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "tt_op": tt_lib_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "tt_op": tt_lib_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "tt_op": tt_lib_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "tt_op": tt_lib_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "tt_op": tt_lib_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "tt_op": tt_lib_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "tt_op": tt_lib_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "tt_op": tt_lib_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "tt_op": tt_lib_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-min-h": {
        "tt_op": tt_lib_ops.reduce_min_h,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-2,)),
    },
    "reduce-min-w": {
        "tt_op": tt_lib_ops.reduce_min_w,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-1,)),
    },
    "reduce-min-hw": {
        "tt_op": tt_lib_ops.reduce_min_hw,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "tt_op": tt_lib_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "tt_op": tt_lib_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "tt_op": tt_lib_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "tt_op": tt_lib_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "tt_op": tt_lib_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
    "transpose-cn": {
        "tt_op": tt_lib_ops.transpose_cn,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=1),
    },
    "transpose-nh": {
        "tt_op": tt_lib_ops.transpose_nh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=-2),
    },
    "transpose-nw": {
        "tt_op": tt_lib_ops.transpose_nw,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=-1),
    },
    "transpose-cw": {
        "tt_op": tt_lib_ops.transpose_cw,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=1, dim1=-1),
    },
    "sum": {
        "tt_op": tt_lib_ops.sum,
        "pytorch_op": pytorch_ops.sum,
    },
    "sum-0": {
        "tt_op": partial(tt_lib_ops.sum, dim=0),
        "pytorch_op": partial(pytorch_ops.sum, dim=0),
    },
    "sum-1": {
        "tt_op": partial(tt_lib_ops.sum, dim=1),
        "pytorch_op": partial(pytorch_ops.sum, dim=1),
    },
    "sum-2": {
        "tt_op": partial(tt_lib_ops.sum, dim=2),
        "pytorch_op": partial(pytorch_ops.sum, dim=2),
    },
    "sum-3": {
        "tt_op": partial(tt_lib_ops.sum, dim=3),
        "pytorch_op": partial(pytorch_ops.sum, dim=3),
    },
    "permute": {
        "tt_op": tt_lib_ops.permute,
        "pytorch_op": pytorch_ops.permute,
    },
    "reshape": {
        "tt_op": tt_lib_ops.reshape,
        "pytorch_op": pytorch_ops.reshape,
    },
    "split-last-dim-two-chunks-tiled": {
        "tt_op": tt_lib_ops.split_last_dim_two_chunks_tiled,
        "pytorch_op": pytorch_ops.split_last_dim_two_chunks_tiled,
    },
    "tilize": {
        "tt_op": tt_lib_ops.tilize,
        "pytorch_op": pytorch_ops.tilize,
    },
    "untilize": {
        "tt_op": tt_lib_ops.untilize,
        "pytorch_op": pytorch_ops.untilize,
    },
    "tilize_with_zero_padding": {
        "tt_op": tt_lib_ops.tilize_with_zero_padding,
        "pytorch_op": pytorch_ops.tilize_with_zero_padding,
    },
    "tilize_with_val_padding": {
        "tt_op": tt_lib_ops.tilize_with_val_padding,
        "pytorch_op": pytorch_ops.tilize_with_val_padding,
    },
    "untilize_with_unpadding": {
        "tt_op": tt_lib_ops.untilize_with_unpadding,
        "pytorch_op": pytorch_ops.untilize_with_unpadding,
    },
    "layernorm": {
        "tt_op": tt_lib_ops.layernorm,
        "pytorch_op": pytorch_ops.layernorm,
    },
    "layernorm-noweights": {
        "tt_op": tt_lib_ops.layernorm_noweights,
        "pytorch_op": pytorch_ops.layernorm_noweights,
    },
    "add-layernorm-noweights": {
        "tt_op": tt_lib_ops.add_layernorm_noweights,
        "pytorch_op": pytorch_ops.add_layernorm_noweights,
    },
    "add-layernorm": {
        "tt_op": tt_lib_ops.add_layernorm,
        "pytorch_op": pytorch_ops.add_layernorm,
    },
    "unpad": {
        "tt_op": tt_lib_ops.unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    ################################################
    #################### Tensor ####################
    ################################################
    "datacopy": {
        "tt_op": tt_lib_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    "tensor_pad": {
        "tt_op": tt_lib_ops.tensor_pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "tensor_unpad": {
        "tt_op": tt_lib_ops.tensor_unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    "pad_to_tile": {
        "tt_op": tt_lib_ops.pad_to_tile,
        "pytorch_op": pytorch_ops.pad_to_tile,
    },
    "unpad_from_tile": {
        "tt_op": tt_lib_ops.unpad_from_tile,
        "pytorch_op": pytorch_ops.unpad_from_tile,
    },
    "conv": {
        "tt_op": tt_lib_ops.conv,
        "pytorch_op": pytorch_ops.conv,
    },
    "repeat_interleave": {
        "tt_op": tt_lib_ops.repeat_interleave,
        "pytorch_op": pytorch_ops.repeat_interleave,
    },
    "repeat": {
        "tt_op": tt_lib_ops.repeat,
        "pytorch_op": pytorch_ops.repeat,
    },
    "activation_glu": {
        "tt_op": tt_lib_ops.activation_glu,
        "pytorch_op": pytorch_ops.activation_glu,
    },
    "activation_reglu": {
        "tt_op": tt_lib_ops.activation_reglu,
        "pytorch_op": pytorch_ops.activation_reglu,
    },
    "activation_geglu": {
        "tt_op": tt_lib_ops.activation_geglu,
        "pytorch_op": pytorch_ops.activation_geglu,
    },
    "activation_swiglu": {
        "tt_op": tt_lib_ops.activation_swiglu,
        "pytorch_op": pytorch_ops.activation_swiglu,
    },
    "bert-large-pre-softmax-bmm": {
        "tt_op": tt_lib_ops.bert_large_pre_softmax_bmm,
        "pytorch_op": pytorch_ops.bert_large_pre_softmax_bmm,
    },
    "bert-large-post-softmax-bmm": {
        "tt_op": tt_lib_ops.bert_large_post_softmax_bmm,
        "pytorch_op": pytorch_ops.bert_large_post_softmax_bmm,
    },
    "bert-large-ff1-matmul": {
        "tt_op": tt_lib_ops.bert_large_ff1_matmul,
        "pytorch_op": pytorch_ops.bert_large_ff1_matmul,
    },
    "bert-large-selfout-matmul": {
        "tt_op": tt_lib_ops.bert_large_selfout_matmul,
        "pytorch_op": pytorch_ops.bert_large_selfout_matmul,
    },
    "bert-large-ff2-matmul": {
        "tt_op": tt_lib_ops.bert_large_ff2_matmul,
        "pytorch_op": pytorch_ops.bert_large_ff2_matmul,
    },
    "embeddings": {
        "tt_op": tt_lib_ops.embeddings,
        "pytorch_op": pytorch_ops.embeddings,
    },
    "embeddings-bw": {
        "tt_op": tt_lib_ops.tt_embedding_bw,
        "pytorch_op": pytorch_ops.pt_embedding_bw,
    },
    "rmsnorm-noweights": {
        "tt_op": tt_lib_ops.rmsnorm_noweights,
        "pytorch_op": pytorch_ops.rmsnorm_noweights,
    },
    "rmsnorm": {
        "tt_op": tt_lib_ops.rmsnorm,
        "pytorch_op": pytorch_ops.rmsnorm,
    },
    "complex-real": {
        "tt_op": tt_lib_ops.complex_real,
        "pytorch_op": pytorch_ops.complex_real,
    },
    "complex-recip": {
        "tt_op": tt_lib_ops.complex_recip,
        "pytorch_op": pytorch_ops.complex_recip,
    },
    "complex-div": {
        "tt_op": tt_lib_ops.complex_div,
        "pytorch_op": pytorch_ops.complex_div,
    },
    "complex-mul": {
        "tt_op": tt_lib_ops.complex_mul,
        "pytorch_op": pytorch_ops.complex_mul,
    },
    "complex-conj": {
        "tt_op": tt_lib_ops.complex_conj,
        "pytorch_op": pytorch_ops.complex_conj,
    },
    "complex-abs": {
        "tt_op": tt_lib_ops.complex_abs,
        "pytorch_op": pytorch_ops.complex_abs,
    },
    "complex-polar": {
        "tt_op": tt_lib_ops.complex_polar,
        "pytorch_op": pytorch_ops.complex_polar,
    },
    "complex-imag": {
        "tt_op": tt_lib_ops.complex_imag,
        "pytorch_op": pytorch_ops.complex_imag,
    },
    "unary-div-bw": {
        "tt_op": tt_lib_ops.unary_div_bw,
        "pytorch_op": pytorch_ops.unary_div_bw,
    },
    "unary-add-bw": {
        "tt_op": tt_lib_ops.unary_add_bw,
        "pytorch_op": pytorch_ops.unary_add_bw,
    },
    "unary-mul-bw": {
        "tt_op": tt_lib_ops.unary_mul_bw,
        "pytorch_op": pytorch_ops.unary_mul_bw,
    },
    "unary-assign-bw": {
        "tt_op": tt_lib_ops.unary_assign_bw,
        "pytorch_op": pytorch_ops.unary_assign_bw,
    },
    "binary-assign-bw": {
        "tt_op": tt_lib_ops.binary_assign_bw,
        "pytorch_op": pytorch_ops.binary_assign_bw,
    },
    "div-bw": {
        "tt_op": tt_lib_ops.div_bw,
        "pytorch_op": pytorch_ops.div_bw,
    },
    "addcdiv-bw": {
        "tt_op": tt_lib_ops.addcdiv_bw,
        "pytorch_op": pytorch_ops.addcdiv_bw,
    },
    "addcmul-bw": {
        "tt_op": tt_lib_ops.addcmul_bw,
        "pytorch_op": pytorch_ops.addcmul_bw,
    },
    "addalpha-bw": {
        "tt_op": tt_lib_ops.addalpha_bw,
        "pytorch_op": pytorch_ops.addalpha_bw,
    },
    "rsqrt-bw": {
        "tt_op": tt_lib_ops.rsqrt_bw,
        "pytorch_op": pytorch_ops.rsqrt_bw,
    },
    "abs-bw": {
        "tt_op": tt_lib_ops.abs_bw,
        "pytorch_op": pytorch_ops.abs_bw,
    },
    "sqrt-bw": {
        "tt_op": tt_lib_ops.sqrt_bw,
        "pytorch_op": pytorch_ops.sqrt_bw,
    },
    "relu-bw": {
        "tt_op": tt_lib_ops.relu_bw,
        "pytorch_op": pytorch_ops.relu_bw,
    },
    "neg-bw": {
        "tt_op": tt_lib_ops.relu_bw,
        "pytorch_op": pytorch_ops.relu_bw,
    },
    "log-bw": {
        "tt_op": tt_lib_ops.log_bw,
        "pytorch_op": pytorch_ops.log_bw,
    },
    "gt-bw": {
        "tt_op": tt_lib_ops.gt_bw,
        "pytorch_op": pytorch_ops.gt_bw,
    },
    "lt-bw": {
        "tt_op": tt_lib_ops.gt_bw,
        "pytorch_op": pytorch_ops.gt_bw,
    },
    "ne-bw": {
        "tt_op": tt_lib_ops.ne_bw,
        "pytorch_op": pytorch_ops.ne_bw,
    },
    "rsub-bw": {
        "tt_op": tt_lib_ops.rsub_bw,
        "pytorch_op": pytorch_ops.rsub_bw,
    },
    "binary-le-bw": {
        "tt_op": tt_lib_ops.binary_le_bw,
        "pytorch_op": pytorch_ops.binary_le_bw,
    },
    "clamp-max-bw": {
        "tt_op": tt_lib_ops.clamp_max_bw,
        "pytorch_op": pytorch_ops.clamp_max_bw,
    },
    "clamp-min-bw": {
        "tt_op": tt_lib_ops.clamp_min_bw,
        "pytorch_op": pytorch_ops.clamp_min_bw,
    },
    "clamp-bw": {
        "tt_op": tt_lib_ops.clamp_bw,
        "pytorch_op": pytorch_ops.clamp_bw,
    },
    "interleaved_to_sharded_partial": {
        "tt_op": tt_lib_ops.interleaved_to_sharded_partial,
        "pytorch_op": pytorch_ops.interleaved_to_sharded_partial,
    },
    "interleaved_to_sharded_partial_coregrid": {
        "tt_op": tt_lib_ops.interleaved_to_sharded_partial_coregrid,
        "pytorch_op": pytorch_ops.interleaved_to_sharded_partial_coregrid,
    },
}
