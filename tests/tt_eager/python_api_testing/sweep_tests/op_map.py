# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


op_map = {
    ################################################
    ################# Helper-Funcs #################
    ################################################
    "linear": {"tt_lib_op": tt_lib_ops.linear, "pytorch_op": pytorch_ops.linear},
    ################################################
    #################### TT-LIB ####################
    ################################################
    "clone": {
        "tt_lib_op": tt_lib_ops.clone,
        "pytorch_op": pytorch_ops.clone,
    },
    "typecast": {
        "tt_lib_op": tt_lib_ops.typecast,
        "pytorch_op": pytorch_ops.typecast,
    },
    "copy": {
        "tt_lib_op": tt_lib_ops.copy,
        "pytorch_op": pytorch_ops.copy,
    },
    "concat": {
        "tt_lib_op": tt_lib_ops.concat,
        "pytorch_op": pytorch_ops.concat,
    },
    "ttnn-concat": {
        "tt_lib_op": ttnn_ops.concat,
        "pytorch_op": pytorch_ops.concat,
    },
    "move": {
        "tt_lib_op": tt_lib_ops.move,
        "pytorch_op": pytorch_ops.move,
    },
    "arange": {
        "tt_lib_op": tt_lib_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    # stats
    "stats-var_hw": {
        "tt_lib_op": tt_lib_ops.var_hw,
        "pytorch_op": pytorch_ops.var_hw,
    },
    "stats-std_hw": {
        "tt_lib_op": tt_lib_ops.std_hw,
        "pytorch_op": pytorch_ops.std_hw,
    },
    "stats-mean_hw": {
        "tt_lib_op": tt_lib_ops.mean_hw,
        "pytorch_op": pytorch_ops.mean_hw,
    },
    "stats-normalize_hw": {
        "tt_lib_op": tt_lib_ops.normalize_hw,
        "pytorch_op": pytorch_ops.normalize_hw,
    },
    "stats-var_global": {
        "tt_lib_op": None,  # tt_lib_ops.var_global,
        "pytorch_op": pytorch_ops.var_global,
    },
    "stats-std_global": {
        "tt_lib_op": None,  # tt_lib_ops.std_global,
        "pytorch_op": pytorch_ops.std_global,
    },
    "stats-mean_global": {
        "tt_lib_op": None,  # tt_lib_ops.mean_global,
        "pytorch_op": pytorch_ops.mean_global,
    },
    "stats-normalize_global": {
        "tt_lib_op": tt_lib_ops.normalize_global,
        "pytorch_op": pytorch_ops.normalize_global,
    },
    # Eltwise unary
    "eltwise-hardtanh": {
        "tt_lib_op": tt_lib_ops.eltwise_hardtanh,
        "pytorch_op": pytorch_ops.hardtanh,
    },
    "eltwise-clip": {
        "tt_lib_op": tt_lib_ops.clip,
        "pytorch_op": pytorch_ops.clip,
    },
    "eltwise-tril": {
        "tt_lib_op": tt_lib_ops.tril,
        "pytorch_op": pytorch_ops.tril,
    },
    "ttnn-tril": {
        "tt_lib_op": ttnn_ops.tril,
        "pytorch_op": pytorch_ops.tril,
    },
    "eltwise-triu": {
        "tt_lib_op": tt_lib_ops.triu,
        "pytorch_op": pytorch_ops.triu,
    },
    "ttnn-triu": {
        "tt_lib_op": ttnn_ops.triu,
        "pytorch_op": pytorch_ops.triu,
    },
    "eltwise-zeros": {
        "tt_lib_op": tt_lib_ops.zeros,
        "pytorch_op": pytorch_ops.zeros,
    },
    "eltwise-empty": {
        "tt_lib_op": tt_lib_ops.empty,
        "pytorch_op": pytorch_ops.empty,
    },
    "eltwise-ones": {
        "tt_lib_op": tt_lib_ops.ones,
        "pytorch_op": pytorch_ops.ones,
    },
    "fill-rm": {
        "tt_lib_op": tt_lib_ops.fill_rm,
        "pytorch_op": pytorch_ops.fill_rm,
    },
    "fill-ones-rm": {
        "tt_lib_op": tt_lib_ops.fill_ones_rm,
        "pytorch_op": pytorch_ops.fill_ones_rm,
    },
    "fill-bw": {
        "tt_lib_op": tt_lib_ops.fill_bw,
        "pytorch_op": pytorch_ops.fill_bw,
    },
    "fill-zero-bw": {
        "tt_lib_op": tt_lib_ops.fill_zero_bw,
        "pytorch_op": pytorch_ops.fill_zero_bw,
    },
    "eltwise-full": {
        "tt_lib_op": tt_lib_ops.full,
        "pytorch_op": pytorch_ops.full,
    },
    "eltwise-zeros_like": {
        "tt_lib_op": tt_lib_ops.zeros_like,
        "pytorch_op": pytorch_ops.zeros_like,
    },
    "eltwise-ones_like": {
        "tt_lib_op": tt_lib_ops.ones_like,
        "pytorch_op": pytorch_ops.ones_like,
    },
    "eltwise-full_like": {
        "tt_lib_op": tt_lib_ops.full_like,
        "pytorch_op": pytorch_ops.full_like,
    },
    "eltwise-div_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_div_unary,
        "pytorch_op": pytorch_ops.div_unary,
    },
    "eltwise-mul_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_mul_unary,
        "pytorch_op": pytorch_ops.mul_unary,
    },
    "eltwise-sub_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_sub_unary,
        "pytorch_op": pytorch_ops.sub_unary,
    },
    "sub-unary-bw": {
        "tt_lib_op": tt_lib_ops.sub_unary_bw,
        "pytorch_op": pytorch_ops.sub_unary_bw,
    },
    "eltwise-add_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_add_unary,
        "pytorch_op": pytorch_ops.add_unary,
    },
    "eltwise-logical_not_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_not_unary,
        "pytorch_op": pytorch_ops.logical_not_unary,
    },
    "eltwise-i0": {
        "tt_lib_op": tt_lib_ops.eltwise_i0,
        "pytorch_op": pytorch_ops.i0,
    },
    "eltwise-lgamma": {
        "tt_lib_op": tt_lib_ops.eltwise_lgamma,
        "pytorch_op": pytorch_ops.lgamma,
    },
    "eltwise-logical_noti": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_noti,
        "pytorch_op": pytorch_ops.logical_noti,
    },
    "eltwise-bitwise_complement": {
        "tt_lib_op": None,  # tt_lib_ops.eltwise_bitwise_complement,
        "pytorch_op": None,  # pytorch_ops.bitwise_complement,
    },
    "eltwise-logical_xor": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_xor,
        "pytorch_op": pytorch_ops.logical_xor,
    },
    "eltwise-sinh": {
        "tt_lib_op": tt_lib_ops.eltwise_sinh,
        "pytorch_op": pytorch_ops.sinh,
    },
    "ttnn-eltwise-sinh": {
        "tt_lib_op": ttnn_ops.eltwise_sinh,
        "pytorch_op": pytorch_ops.sinh,
    },
    "eltwise-cosh": {
        "tt_lib_op": tt_lib_ops.eltwise_cosh,
        "pytorch_op": pytorch_ops.cosh,
    },
    "eltwise-ltz": {
        "tt_lib_op": tt_lib_ops.eltwise_ltz,
        "pytorch_op": pytorch_ops.ltz,
    },
    "eltwise-gtz": {
        "tt_lib_op": tt_lib_ops.eltwise_gtz,
        "pytorch_op": pytorch_ops.gtz,
    },
    "eltwise-lez": {
        "tt_lib_op": tt_lib_ops.eltwise_lez,
        "pytorch_op": pytorch_ops.lez,
    },
    "eltwise-gez": {
        "tt_lib_op": tt_lib_ops.eltwise_gez,
        "pytorch_op": pytorch_ops.gez,
    },
    "eltwise-eqz": {
        "tt_lib_op": tt_lib_ops.eltwise_eqz,
        "pytorch_op": pytorch_ops.eqz,
    },
    "ttnn-eltwise-eqz": {
        "tt_lib_op": ttnn_ops.eltwise_eqz,
        "pytorch_op": pytorch_ops.eqz,
    },
    "eltwise-nez": {
        "tt_lib_op": tt_lib_ops.eltwise_nez,
        "pytorch_op": pytorch_ops.nez,
    },
    "eltwise-abs": {
        "tt_lib_op": tt_lib_ops.eltwise_abs,
        "pytorch_op": pytorch_ops.abs,
    },
    "eltwise-digamma": {
        "tt_lib_op": tt_lib_ops.eltwise_digamma,
        "pytorch_op": pytorch_ops.digamma,
    },
    "eltwise-isfinite": {
        "tt_lib_op": tt_lib_ops.eltwise_isfinite,
        "pytorch_op": pytorch_ops.isfinite,
    },
    "eltwise-isinf": {
        "tt_lib_op": tt_lib_ops.eltwise_isinf,
        "pytorch_op": pytorch_ops.isinf,
    },
    "eltwise-isposinf": {
        "tt_lib_op": tt_lib_ops.eltwise_isposinf,
        "pytorch_op": pytorch_ops.isposinf,
    },
    "eltwise-isneginf": {
        "tt_lib_op": tt_lib_ops.eltwise_isneginf,
        "pytorch_op": pytorch_ops.isneginf,
    },
    "eltwise-isnan": {
        "tt_lib_op": tt_lib_ops.eltwise_isnan,
        "pytorch_op": pytorch_ops.isnan,
    },
    "eltwise-sign": {
        "tt_lib_op": tt_lib_ops.eltwise_sign,
        "pytorch_op": pytorch_ops.sign,
    },
    "ttnn-eltwise-sign": {
        "tt_lib_op": ttnn_ops.eltwise_sign,
        "pytorch_op": pytorch_ops.sign,
    },
    "eltwise-multigammaln": {
        "tt_lib_op": tt_lib_ops.eltwise_multigammaln,
        "pytorch_op": pytorch_ops.multigammaln,
    },
    "eltwise-silu": {
        "tt_lib_op": tt_lib_ops.eltwise_silu,
        "pytorch_op": pytorch_ops.silu,
    },
    "ttnn-eltwise-silu": {
        "tt_lib_op": ttnn_ops.eltwise_silu,
        "pytorch_op": pytorch_ops.silu,
    },
    "eltwise-elu": {
        "tt_lib_op": tt_lib_ops.eltwise_elu,
        "pytorch_op": pytorch_ops.elu,
    },
    "eltwise-square": {
        "tt_lib_op": tt_lib_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "ttnn-eltwise-square": {
        "tt_lib_op": ttnn_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "eltwise-mish": {
        "tt_lib_op": tt_lib_ops.eltwise_mish,
        "pytorch_op": pytorch_ops.mish,
    },
    "eltwise-softplus": {
        "tt_lib_op": tt_lib_ops.eltwise_softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "ttnn-eltwise-softplus": {
        "tt_lib_op": ttnn_ops.eltwise_softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "eltwise-log1p": {
        "tt_lib_op": tt_lib_ops.eltwise_log1p,
        "pytorch_op": pytorch_ops.log1p,
    },
    "eltwise-add1": {
        "tt_lib_op": tt_lib_ops.eltwise_add1,
        "pytorch_op": pytorch_ops.add1,
    },
    "eltwise-neg": {
        "tt_lib_op": tt_lib_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
    },
    "eltwise-swish": {
        "tt_lib_op": tt_lib_ops.eltwise_swish,
        "pytorch_op": pytorch_ops.swish,
    },
    "ttnn-eltwise-swish": {
        "tt_lib_op": ttnn_ops.eltwise_swish,
        "pytorch_op": pytorch_ops.swish,
    },
    "eltwise-cos": {
        "tt_lib_op": tt_lib_ops.eltwise_cos,
        "pytorch_op": pytorch_ops.cos,
    },
    "eltwise-sin": {
        "tt_lib_op": tt_lib_ops.eltwise_sin,
        "pytorch_op": pytorch_ops.sin,
    },
    "ttnn-eltwise-sin": {
        "tt_lib_op": ttnn_ops.eltwise_sin,
        "pytorch_op": pytorch_ops.sin,
    },
    "eltwise-tan": {
        "tt_lib_op": tt_lib_ops.eltwise_tan,
        "pytorch_op": pytorch_ops.tan,
    },
    "ttnn-eltwise-tan": {
        "tt_lib_op": ttnn_ops.eltwise_tan,
        "pytorch_op": pytorch_ops.tan,
    },
    "eltwise-tan-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_tan_bw,
        "pytorch_op": pytorch_ops.tan_bw,
    },
    "eltwise-asin": {
        "tt_lib_op": tt_lib_ops.eltwise_asin,
        "pytorch_op": pytorch_ops.asin,
    },
    "eltwise-atan": {
        "tt_lib_op": tt_lib_ops.eltwise_atan,
        "pytorch_op": pytorch_ops.atan,
    },
    "eltwise-atanh": {
        "tt_lib_op": tt_lib_ops.eltwise_atanh,
        "pytorch_op": pytorch_ops.atanh,
    },
    "eltwise-acos": {
        "tt_lib_op": tt_lib_ops.eltwise_acos,
        "pytorch_op": pytorch_ops.acos,
    },
    "eltwise-asinh": {
        "tt_lib_op": tt_lib_ops.eltwise_asinh,
        "pytorch_op": pytorch_ops.asinh,
    },
    "eltwise-acosh": {
        "tt_lib_op": tt_lib_ops.eltwise_acosh,
        "pytorch_op": pytorch_ops.acosh,
    },
    "eltwise-exp": {
        "tt_lib_op": tt_lib_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-exp-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_exp_bw,
        "pytorch_op": pytorch_ops.exp_bw,
    },
    "eltwise-exp2": {
        "tt_lib_op": tt_lib_ops.eltwise_exp2,
        "pytorch_op": pytorch_ops.exp2,
    },
    "eltwise-expm1": {
        "tt_lib_op": tt_lib_ops.eltwise_expm1,
        "pytorch_op": pytorch_ops.expm1,
    },
    "eltwise-recip": {
        "tt_lib_op": tt_lib_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "ttnn-eltwise-recip": {
        "tt_lib_op": ttnn_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "tt_lib_op": tt_lib_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "ttnn-eltwise-sqrt": {
        "tt_lib_op": ttnn_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "tt_lib_op": tt_lib_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-softmax_in_place": {
        "tt_lib_op": tt_lib_ops.eltwise_softmax_in_place,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "eltwise-scale_mask_softmax_in_place": {
        "tt_lib_op": tt_lib_ops.eltwise_scale_mask_softmax_in_place,
        "pytorch_op": pytorch_ops.scale_mask_softmax_in_place,
    },
    "eltwise-rsqrt": {
        "tt_lib_op": tt_lib_ops.eltwise_rsqrt,
        "pytorch_op": pytorch_ops.rsqrt,
    },
    "ttnn-rsqrt": {
        "tt_lib_op": ttnn_ops.eltwise_rsqrt,
        "pytorch_op": pytorch_ops.rsqrt,
    },
    "eltwise-xlogy": {
        "tt_lib_op": tt_lib_ops.eltwise_xlogy,
        "pytorch_op": pytorch_ops.xlogy,
    },
    "eltwise-logical_and": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_and,
        "pytorch_op": pytorch_ops.logical_and,
    },
    "eltwise-logical_andi": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_andi,
        "pytorch_op": pytorch_ops.logical_andi,
    },
    "eltwise-atan2": {
        "tt_lib_op": tt_lib_ops.eltwise_atan2,
        "pytorch_op": pytorch_ops.atan2,
    },
    "eltwise-lerp_binary": {
        "tt_lib_op": tt_lib_ops.eltwise_lerp_binary,
        "pytorch_op": pytorch_ops.lerp_binary,
    },
    "eltwise-lerp_ternary": {
        "tt_lib_op": tt_lib_ops.eltwise_lerp_ternary,
        "pytorch_op": pytorch_ops.lerp_ternary,
    },
    "eltwise-leaky_relu": {
        "tt_lib_op": tt_lib_ops.eltwise_leaky_relu,
        "pytorch_op": pytorch_ops.leaky_relu,
    },
    "eltwise-prelu": {
        "tt_lib_op": tt_lib_ops.eltwise_prelu,
        "pytorch_op": pytorch_ops.prelu,
    },
    "eltwise-hardshrink": {
        "tt_lib_op": tt_lib_ops.eltwise_hardshrink,
        "pytorch_op": pytorch_ops.hardshrink,
    },
    "eltwise-bias_gelu_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_bias_gelu_unary,
        "pytorch_op": pytorch_ops.bias_gelu_unary,
    },
    "eltwise-softshrink": {
        "tt_lib_op": tt_lib_ops.eltwise_softshrink,
        "pytorch_op": pytorch_ops.softshrink,
    },
    "ttnn-eltwise-softshrink": {
        "tt_lib_op": ttnn_ops.eltwise_softshrink,
        "pytorch_op": pytorch_ops.softshrink,
    },
    "eltwise-softsign": {
        "tt_lib_op": tt_lib_ops.eltwise_softsign,
        "pytorch_op": pytorch_ops.softsign,
    },
    "ttnn-eltwise-softsign": {
        "tt_lib_op": ttnn_ops.eltwise_softsign,
        "pytorch_op": pytorch_ops.softsign,
    },
    "eltwise-relu": {
        "tt_lib_op": tt_lib_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "eltwise-pow": {
        "tt_lib_op": tt_lib_ops.eltwise_pow,
        "pytorch_op": pytorch_ops.power,
    },
    "unary-pow-bw": {
        "tt_lib_op": tt_lib_ops.unary_pow_bw,
        "pytorch_op": pytorch_ops.power_bw,
    },
    "bert-large-fused-qkv-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_fused_qkv_matmul,
        "pytorch_op": pytorch_ops.bert_large_fused_qkv_matmul,
    },
    "eltwise-relu_max": {
        "tt_lib_op": tt_lib_ops.eltwise_relu_max,
        "pytorch_op": pytorch_ops.relu_max,
    },
    "eltwise-relu_min": {
        "tt_lib_op": tt_lib_ops.eltwise_relu_min,
        "pytorch_op": pytorch_ops.relu_min,
    },
    "eltwise-polyval": {
        "tt_lib_op": tt_lib_ops.eltwise_polyval,
        "pytorch_op": pytorch_ops.polyval,
    },
    "ttnn-eltwise-polyval": {
        "tt_lib_op": ttnn_ops.eltwise_polyval,
        "pytorch_op": pytorch_ops.polyval,
    },
    "eltwise-mac": {
        "tt_lib_op": tt_lib_ops.eltwise_mac,
        "pytorch_op": pytorch_ops.mac,
    },
    "eltwise-addcmul": {
        "tt_lib_op": tt_lib_ops.eltwise_addcmul,
        "pytorch_op": pytorch_ops.addcmul,
    },
    "eltwise-addcdiv": {
        "tt_lib_op": tt_lib_ops.eltwise_addcdiv,
        "pytorch_op": pytorch_ops.addcdiv,
    },
    "ttnn-eltwise-addcdiv": {
        "tt_lib_op": ttnn_ops.eltwise_addcdiv,
        "pytorch_op": pytorch_ops.addcdiv,
    },
    "eltwise-sigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "ttnn-eltwise-sigmoid": {
        "tt_lib_op": ttnn_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-log_sigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_log_sigmoid,
        "pytorch_op": pytorch_ops.log_sigmoid,
    },
    "eltwise-heaviside": {
        "tt_lib_op": tt_lib_ops.eltwise_heaviside,
        "pytorch_op": pytorch_ops.heaviside,
    },
    "eltwise-erf": {
        "tt_lib_op": tt_lib_ops.eltwise_erf,
        "pytorch_op": pytorch_ops.erf,
    },
    "eltwise-erfc": {
        "tt_lib_op": tt_lib_ops.eltwise_erfc,
        "pytorch_op": pytorch_ops.erfc,
    },
    "eltwise-erfinv": {
        "tt_lib_op": tt_lib_ops.eltwise_erfinv,
        "pytorch_op": pytorch_ops.erfinv,
    },
    "eltwise-nextafter": {
        "tt_lib_op": tt_lib_ops.eltwise_nextafter,
        "pytorch_op": pytorch_ops.nextafter,
    },
    "eltwise-subalpha": {
        "tt_lib_op": tt_lib_ops.eltwise_subalpha,
        "pytorch_op": pytorch_ops.subalpha,
    },
    "eltwise-addalpha": {
        "tt_lib_op": tt_lib_ops.eltwise_addalpha,
        "pytorch_op": pytorch_ops.addalpha,
    },
    "lamb-optimizer": {
        "tt_lib_op": tt_lib_ops.lamb_optimizer,
        "pytorch_op": pytorch_ops.lamb_optimizer,
    },
    "eltwise-logit": {
        "tt_lib_op": tt_lib_ops.eltwise_logit,
        "pytorch_op": pytorch_ops.logit,
    },
    "eltwise-polygamma": {
        "tt_lib_op": tt_lib_ops.eltwise_polygamma,
        "pytorch_op": pytorch_ops.polygamma,
    },
    "ttnn-eltwise-polygamma": {
        "tt_lib_op": ttnn_ops.eltwise_polygamma,
        "pytorch_op": pytorch_ops.polygamma,
    },
    "eltwise-logical_xori": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_xori,
        "pytorch_op": pytorch_ops.logical_xori,
    },
    "eltwise-hardsigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_hardsigmoid,
        "pytorch_op": pytorch_ops.hardsigmoid,
    },
    "eltwise-hardswish": {
        "tt_lib_op": tt_lib_ops.eltwise_hardswish,
        "pytorch_op": pytorch_ops.hardswish,
    },
    "eltwise-log": {
        "tt_lib_op": tt_lib_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "eltwise-log2": {
        "tt_lib_op": tt_lib_ops.eltwise_log2,
        "pytorch_op": pytorch_ops.log2,
    },
    "eltwise-log10": {
        "tt_lib_op": tt_lib_ops.eltwise_log10,
        "pytorch_op": pytorch_ops.log10,
    },
    "eltwise-tanh": {
        "tt_lib_op": tt_lib_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    "eltwise-tanh-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_tanh_bw,
        "pytorch_op": pytorch_ops.tanh_bw,
    },
    "eltwise-tanhshrink": {
        "tt_lib_op": tt_lib_ops.eltwise_tanhshrink,
        "pytorch_op": pytorch_ops.tanhshrink,
    },
    "ttnn-eltwise-tanhshrink": {
        "tt_lib_op": ttnn_ops.eltwise_tanhshrink,
        "pytorch_op": pytorch_ops.tanhshrink,
    },
    "eltwise-signbit": {
        "tt_lib_op": tt_lib_ops.eltwise_signbit,
        "pytorch_op": pytorch_ops.signbit,
    },
    "ttnn-eltwise-signbit": {
        "tt_lib_op": ttnn_ops.eltwise_signbit,
        "pytorch_op": pytorch_ops.signbit,
    },
    "eltwise-rpow": {
        "tt_lib_op": tt_lib_ops.eltwise_rpow,
        "pytorch_op": pytorch_ops.eltwise_rpow,
    },
    "eltwise-rdiv": {
        "tt_lib_op": tt_lib_ops.eltwise_rdiv,
        "pytorch_op": pytorch_ops.eltwise_rdiv,
    },
    "eltwise-rsub": {
        "tt_lib_op": tt_lib_ops.eltwise_rsub,
        "pytorch_op": pytorch_ops.eltwise_rsub,
    },
    "eltwise-identity": {
        "tt_lib_op": tt_lib_ops.eltwise_identity,
        "pytorch_op": pytorch_ops.eltwise_identity,
    },
    # Eltwise binary
    "eltwise-ne": {
        "tt_lib_op": tt_lib_ops.eltwise_ne,
        "pytorch_op": pytorch_ops.ne,
    },
    "ttnn-eltwise-ne": {
        "tt_lib_op": ttnn_ops.eltwise_ne,
        "pytorch_op": pytorch_ops.ne,
    },
    "eltwise-bias_gelu": {
        "tt_lib_op": tt_lib_ops.eltwise_bias_gelu,
        "pytorch_op": pytorch_ops.bias_gelu,
    },
    "eltwise-eq": {
        "tt_lib_op": tt_lib_ops.eltwise_eq,
        "pytorch_op": pytorch_ops.eq,
    },
    "ttnn-eltwise-eq": {
        "tt_lib_op": ttnn_ops.eltwise_eq,
        "pytorch_op": pytorch_ops.eq,
    },
    "eltwise-lt": {
        "tt_lib_op": tt_lib_ops.eltwise_lt,
        "pytorch_op": pytorch_ops.lt,
    },
    "ttnn-eltwise-lt": {
        "tt_lib_op": ttnn_ops.eltwise_lt,
        "pytorch_op": pytorch_ops.lt,
    },
    "eltwise-gt": {
        "tt_lib_op": tt_lib_ops.eltwise_gt,
        "pytorch_op": pytorch_ops.gt,
    },
    "ttnn-eltwise-gt": {
        "tt_lib_op": ttnn_ops.eltwise_gt,
        "pytorch_op": pytorch_ops.gt,
    },
    "eltwise-gte": {
        "tt_lib_op": tt_lib_ops.eltwise_gte,
        "pytorch_op": pytorch_ops.gte,
    },
    "ttnn-eltwise-gte": {
        "tt_lib_op": ttnn_ops.eltwise_gte,
        "pytorch_op": pytorch_ops.gte,
    },
    "eltwise-lte": {
        "tt_lib_op": tt_lib_ops.eltwise_lte,
        "pytorch_op": pytorch_ops.lte,
    },
    "ttnn-eltwise-lte": {
        "tt_lib_op": ttnn_ops.eltwise_lte,
        "pytorch_op": pytorch_ops.lte,
    },
    "eltwise-add": {
        "tt_lib_op": tt_lib_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-add-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_add_bw,
        "pytorch_op": pytorch_ops.add_bw,
    },
    "eltwise-sub": {
        "tt_lib_op": tt_lib_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-sub-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_sub_bw,
        "pytorch_op": pytorch_ops.sub_bw,
    },
    "eltwise-mul": {
        "tt_lib_op": tt_lib_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    "eltwise-mul-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_mul_bw,
        "pytorch_op": pytorch_ops.mul_bw,
    },
    "ttnn-eltwise-minimum": {
        "tt_lib_op": ttnn_ops.eltwise_minimum,
        "pytorch_op": pytorch_ops.minimum,
    },
    "eltwise-min": {
        "tt_lib_op": tt_lib_ops.eltwise_min,
        "pytorch_op": pytorch_ops.min,
    },
    "eltwise-min-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_min_bw,
        "pytorch_op": pytorch_ops.min_bw,
    },
    "eltwise-max": {
        "tt_lib_op": tt_lib_ops.eltwise_max,
        "pytorch_op": pytorch_ops.max,
    },
    "eltwise-max-bw": {
        "tt_lib_op": tt_lib_ops.eltwise_max_bw,
        "pytorch_op": pytorch_ops.max_bw,
    },
    "eltwise-squared_difference": {
        "tt_lib_op": tt_lib_ops.eltwise_squared_difference,
        "pytorch_op": pytorch_ops.squared_difference,
    },
    "eltwise-deg2rad": {
        "tt_lib_op": tt_lib_ops.eltwise_deg2rad,
        "pytorch_op": pytorch_ops.deg2rad,
    },
    "eltwise-rad2deg": {
        "tt_lib_op": tt_lib_ops.eltwise_rad2deg,
        "pytorch_op": pytorch_ops.rad2deg,
    },
    "ttnn-eltwise-rad2deg": {
        "tt_lib_op": ttnn_ops.eltwise_rad2deg,
        "pytorch_op": pytorch_ops.rad2deg,
    },
    "eltwise-cbrt": {
        "tt_lib_op": tt_lib_ops.eltwise_cbrt,
        "pytorch_op": pytorch_ops.cbrt,
    },
    "eltwise-hypot": {
        "tt_lib_op": tt_lib_ops.eltwise_hypot,
        "pytorch_op": pytorch_ops.hypot,
    },
    "eltwise-scatter": {
        "tt_lib_op": tt_lib_ops.eltwise_scatter,
        "pytorch_op": pytorch_ops.scatter,
    },
    "eltwise-threshold": {
        "tt_lib_op": tt_lib_ops.eltwise_threshold,
        "pytorch_op": pytorch_ops.threshold,
    },
    "ttnn-eltwise-threshold": {
        "tt_lib_op": ttnn_ops.eltwise_threshold,
        "pytorch_op": pytorch_ops.threshold,
    },
    "eltwise-relu6": {
        "tt_lib_op": tt_lib_ops.eltwise_relu6,
        "pytorch_op": pytorch_ops.relu6,
    },
    "ttnn-relu6": {
        "tt_lib_op": ttnn_ops.eltwise_relu6,
        "pytorch_op": pytorch_ops.relu6,
    },
    "eltwise-ldexp": {
        "tt_lib_op": tt_lib_ops.eltwise_ldexp,
        "pytorch_op": pytorch_ops.ldexp,
    },
    "eltwise-logaddexp": {
        "tt_lib_op": tt_lib_ops.eltwise_logaddexp,
        "pytorch_op": pytorch_ops.logaddexp,
    },
    "eltwise-logaddexp2": {
        "tt_lib_op": tt_lib_ops.eltwise_logaddexp2,
        "pytorch_op": pytorch_ops.logaddexp2,
    },
    "eltwise-assign_binary": {
        "tt_lib_op": tt_lib_ops.eltwise_assign_binary,
        "pytorch_op": pytorch_ops.assign_binary,
    },
    "eltwise-assign_unary": {
        "tt_lib_op": tt_lib_ops.eltwise_assign_unary,
        "pytorch_op": pytorch_ops.assign_unary,
    },
    "eltwise-logical_or": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_or,
        "pytorch_op": pytorch_ops.logical_or,
    },
    "eltwise-logical_ori": {
        "tt_lib_op": tt_lib_ops.eltwise_logical_ori,
        "pytorch_op": pytorch_ops.logical_ori,
    },
    "eltwise-isclose": {
        "tt_lib_op": tt_lib_ops.eltwise_isclose,
        "pytorch_op": pytorch_ops.isclose,
    },
    "ttnn-eltwise-isclose": {
        "tt_lib_op": ttnn_ops.eltwise_isclose,
        "pytorch_op": pytorch_ops.isclose,
    },
    # Eltwise ternary
    "eltwise-arange": {
        "tt_lib_op": tt_lib_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    "eltwise-where": {
        "tt_lib_op": tt_lib_ops.where,
        "pytorch_op": pytorch_ops.where,
    },
    "ttnn-eltwise-where": {
        "tt_lib_op": ttnn_ops.where,
        "pytorch_op": pytorch_ops.where,
    },
    "where-bw": {
        "tt_lib_op": tt_lib_ops.where_bw,
        "pytorch_op": pytorch_ops.where_bw,
    },
    # Matmul
    "matmul": {
        "tt_lib_op": tt_lib_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "outer": {
        "tt_lib_op": tt_lib_ops.outer,
        "pytorch_op": pytorch_ops.outer,
    },
    "bmm": {
        "tt_lib_op": tt_lib_ops.bmm,
        "pytorch_op": pytorch_ops.matmul,
    },
    # Broadcast
    "bcast-add-h": {
        "tt_lib_op": tt_lib_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "tt_lib_op": tt_lib_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "tt_lib_op": tt_lib_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "tt_lib_op": tt_lib_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "tt_lib_op": tt_lib_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "tt_lib_op": tt_lib_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "tt_lib_op": tt_lib_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "tt_lib_op": tt_lib_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "tt_lib_op": tt_lib_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "tt_lib_op": tt_lib_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "tt_lib_op": tt_lib_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "tt_lib_op": tt_lib_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-min-h": {
        "tt_lib_op": tt_lib_ops.reduce_min_h,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-2,)),
    },
    "reduce-min-w": {
        "tt_lib_op": tt_lib_ops.reduce_min_w,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-1,)),
    },
    "reduce-min-hw": {
        "tt_lib_op": tt_lib_ops.reduce_min_hw,
        "pytorch_op": partial(pytorch_ops.reduce_min, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "tt_lib_op": tt_lib_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "tt_lib_op": tt_lib_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "tt_lib_op": tt_lib_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "tt_lib_op": tt_lib_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "tt_lib_op": tt_lib_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
    "transpose-cn": {
        "tt_lib_op": tt_lib_ops.transpose_cn,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=1),
    },
    "transpose-nh": {
        "tt_lib_op": tt_lib_ops.transpose_nh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=-2),
    },
    "transpose-nw": {
        "tt_lib_op": tt_lib_ops.transpose_nw,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=-1),
    },
    "transpose-cw": {
        "tt_lib_op": tt_lib_ops.transpose_cw,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=1, dim1=-1),
    },
    "sum": {
        "tt_lib_op": tt_lib_ops.sum,
        "pytorch_op": pytorch_ops.sum,
    },
    "ttnn-sum": {
        "tt_lib_op": ttnn_ops.sum,
        "pytorch_op": pytorch_ops.sum,
    },
    "sum-0": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=0),
        "pytorch_op": partial(pytorch_ops.sum, dim=0),
    },
    "sum-1": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=1),
        "pytorch_op": partial(pytorch_ops.sum, dim=1),
    },
    "sum-2": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=2),
        "pytorch_op": partial(pytorch_ops.sum, dim=2),
    },
    "sum-3": {
        "tt_lib_op": partial(tt_lib_ops.sum, dim=3),
        "pytorch_op": partial(pytorch_ops.sum, dim=3),
    },
    "permute": {
        "tt_lib_op": tt_lib_ops.permute,
        "pytorch_op": pytorch_ops.permute,
    },
    "reshape": {
        "tt_lib_op": tt_lib_ops.reshape,
        "pytorch_op": pytorch_ops.reshape,
    },
    "split-last-dim-two-chunks-tiled": {
        "tt_lib_op": tt_lib_ops.split_last_dim_two_chunks_tiled,
        "pytorch_op": pytorch_ops.split_last_dim_two_chunks_tiled,
    },
    "tilize": {
        "tt_lib_op": tt_lib_ops.tilize,
        "pytorch_op": pytorch_ops.tilize,
    },
    "untilize": {
        "tt_lib_op": tt_lib_ops.untilize,
        "pytorch_op": pytorch_ops.untilize,
    },
    "tilize_with_zero_padding": {
        "tt_lib_op": tt_lib_ops.tilize_with_zero_padding,
        "pytorch_op": pytorch_ops.tilize_with_zero_padding,
    },
    "tilize_with_val_padding": {
        "tt_lib_op": tt_lib_ops.tilize_with_val_padding,
        "pytorch_op": pytorch_ops.tilize_with_val_padding,
    },
    "untilize_with_unpadding": {
        "tt_lib_op": tt_lib_ops.untilize_with_unpadding,
        "pytorch_op": pytorch_ops.untilize_with_unpadding,
    },
    "layernorm": {
        "tt_lib_op": tt_lib_ops.layernorm,
        "pytorch_op": pytorch_ops.layernorm,
    },
    "layernorm-noweights": {
        "tt_lib_op": tt_lib_ops.layernorm_noweights,
        "pytorch_op": pytorch_ops.layernorm_noweights,
    },
    "add-layernorm-noweights": {
        "tt_lib_op": tt_lib_ops.add_layernorm_noweights,
        "pytorch_op": pytorch_ops.add_layernorm_noweights,
    },
    "add-layernorm": {
        "tt_lib_op": tt_lib_ops.add_layernorm,
        "pytorch_op": pytorch_ops.add_layernorm,
    },
    "pad": {
        "tt_lib_op": tt_lib_ops.pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "unpad": {
        "tt_lib_op": tt_lib_ops.unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    ################################################
    #################### Tensor ####################
    ################################################
    "datacopy": {
        "tt_lib_op": tt_lib_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    "tensor_pad": {
        "tt_lib_op": tt_lib_ops.tensor_pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "tensor_unpad": {
        "tt_lib_op": tt_lib_ops.tensor_unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    "pad_to_tile": {
        "tt_lib_op": tt_lib_ops.pad_to_tile,
        "pytorch_op": pytorch_ops.pad_to_tile,
    },
    "unpad_from_tile": {
        "tt_lib_op": tt_lib_ops.unpad_from_tile,
        "pytorch_op": pytorch_ops.unpad_from_tile,
    },
    "conv": {
        "tt_lib_op": tt_lib_ops.conv,
        "pytorch_op": pytorch_ops.conv,
    },
    "repeat_interleave": {
        "tt_lib_op": tt_lib_ops.repeat_interleave,
        "pytorch_op": pytorch_ops.repeat_interleave,
    },
    "repeat": {
        "tt_lib_op": tt_lib_ops.repeat,
        "pytorch_op": pytorch_ops.repeat,
    },
    "activation_glu": {
        "tt_lib_op": tt_lib_ops.activation_glu,
        "pytorch_op": pytorch_ops.activation_glu,
    },
    "ttnn-activation_glu": {
        "tt_lib_op": ttnn_ops.activation_glu,
        "pytorch_op": pytorch_ops.activation_glu,
    },
    "activation_reglu": {
        "tt_lib_op": tt_lib_ops.activation_reglu,
        "pytorch_op": pytorch_ops.activation_reglu,
    },
    "activation_geglu": {
        "tt_lib_op": tt_lib_ops.activation_geglu,
        "pytorch_op": pytorch_ops.activation_geglu,
    },
    "activation_swiglu": {
        "tt_lib_op": tt_lib_ops.activation_swiglu,
        "pytorch_op": pytorch_ops.activation_swiglu,
    },
    "groupnorm-noweights": {
        "tt_lib_op": tt_lib_ops.groupnorm_noweights,
        "pytorch_op": pytorch_ops.groupnorm_noweights,
    },
    "bert-large-pre-softmax-bmm": {
        "tt_lib_op": tt_lib_ops.bert_large_pre_softmax_bmm,
        "pytorch_op": pytorch_ops.bert_large_pre_softmax_bmm,
    },
    "bert-large-post-softmax-bmm": {
        "tt_lib_op": tt_lib_ops.bert_large_post_softmax_bmm,
        "pytorch_op": pytorch_ops.bert_large_post_softmax_bmm,
    },
    "bert-large-ff1-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_ff1_matmul,
        "pytorch_op": pytorch_ops.bert_large_ff1_matmul,
    },
    "bert-large-selfout-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_selfout_matmul,
        "pytorch_op": pytorch_ops.bert_large_selfout_matmul,
    },
    "bert-large-ff2-matmul": {
        "tt_lib_op": tt_lib_ops.bert_large_ff2_matmul,
        "pytorch_op": pytorch_ops.bert_large_ff2_matmul,
    },
    "embeddings": {
        "tt_lib_op": tt_lib_ops.embeddings,
        "pytorch_op": pytorch_ops.embeddings,
    },
    "embeddings-bw": {
        "tt_lib_op": tt_lib_ops.tt_embedding_bw,
        "pytorch_op": pytorch_ops.pt_embedding_bw,
    },
    "rmsnorm-noweights": {
        "tt_lib_op": tt_lib_ops.rmsnorm_noweights,
        "pytorch_op": pytorch_ops.rmsnorm_noweights,
    },
    "rmsnorm": {
        "tt_lib_op": tt_lib_ops.rmsnorm,
        "pytorch_op": pytorch_ops.rmsnorm,
    },
    "groupnorm": {
        "tt_lib_op": tt_lib_ops.groupnorm,
        "pytorch_op": pytorch_ops.groupnorm,
    },
    "complex-real": {
        "tt_lib_op": tt_lib_ops.complex_real,
        "pytorch_op": pytorch_ops.complex_real,
    },
    "complex-recip": {
        "tt_lib_op": tt_lib_ops.complex_recip,
        "pytorch_op": pytorch_ops.complex_recip,
    },
    "complex-div": {
        "tt_lib_op": tt_lib_ops.complex_div,
        "pytorch_op": pytorch_ops.complex_div,
    },
    "complex-mul": {
        "tt_lib_op": tt_lib_ops.complex_mul,
        "pytorch_op": pytorch_ops.complex_mul,
    },
    "complex-conj": {
        "tt_lib_op": tt_lib_ops.complex_conj,
        "pytorch_op": pytorch_ops.complex_conj,
    },
    "complex-abs": {
        "tt_lib_op": tt_lib_ops.complex_abs,
        "pytorch_op": pytorch_ops.complex_abs,
    },
    "complex-polar": {
        "tt_lib_op": tt_lib_ops.complex_polar,
        "pytorch_op": pytorch_ops.complex_polar,
    },
    "complex-imag": {
        "tt_lib_op": tt_lib_ops.complex_imag,
        "pytorch_op": pytorch_ops.complex_imag,
    },
    "unary-div-bw": {
        "tt_lib_op": tt_lib_ops.unary_div_bw,
        "pytorch_op": pytorch_ops.unary_div_bw,
    },
    "unary-add-bw": {
        "tt_lib_op": tt_lib_ops.unary_add_bw,
        "pytorch_op": pytorch_ops.unary_add_bw,
    },
    "unary-mul-bw": {
        "tt_lib_op": tt_lib_ops.unary_mul_bw,
        "pytorch_op": pytorch_ops.unary_mul_bw,
    },
    "unary-assign-bw": {
        "tt_lib_op": tt_lib_ops.unary_assign_bw,
        "pytorch_op": pytorch_ops.unary_assign_bw,
    },
    "binary-assign-bw": {
        "tt_lib_op": tt_lib_ops.binary_assign_bw,
        "pytorch_op": pytorch_ops.binary_assign_bw,
    },
    "div-bw": {
        "tt_lib_op": tt_lib_ops.div_bw,
        "pytorch_op": pytorch_ops.div_bw,
    },
    "addcdiv-bw": {
        "tt_lib_op": tt_lib_ops.addcdiv_bw,
        "pytorch_op": pytorch_ops.addcdiv_bw,
    },
    "addcmul-bw": {
        "tt_lib_op": tt_lib_ops.addcmul_bw,
        "pytorch_op": pytorch_ops.addcmul_bw,
    },
    "addalpha-bw": {
        "tt_lib_op": tt_lib_ops.addalpha_bw,
        "pytorch_op": pytorch_ops.addalpha_bw,
    },
    "rsqrt-bw": {
        "tt_lib_op": tt_lib_ops.rsqrt_bw,
        "pytorch_op": pytorch_ops.rsqrt_bw,
    },
    "ttnn-eltwise-ones": {
        "tt_lib_op": ttnn_ops.ones,
        "pytorch_op": pytorch_ops.ones,
    },
    "ttnn-eltwise-ones_like": {
        "tt_lib_op": ttnn_ops.ones_like,
        "pytorch_op": pytorch_ops.ones_like,
    },
    "ttnn-eltwise-full": {
        "tt_lib_op": ttnn_ops.full,
        "pytorch_op": pytorch_ops.full,
    },
    "ttnn-eltwise-hardswish": {
        "tt_lib_op": ttnn_ops.eltwise_hardswish,
        "pytorch_op": pytorch_ops.hardswish,
    },
    "ttnn-eltwise-hardtanh": {
        "tt_lib_op": ttnn_ops.eltwise_hardtanh,
        "pytorch_op": pytorch_ops.hardtanh,
    },
    "ttnn-eltwise-heaviside": {
        "tt_lib_op": ttnn_ops.eltwise_heaviside,
        "pytorch_op": pytorch_ops.heaviside,
    },
    "ttnn-eltwise-hypot": {
        "tt_lib_op": ttnn_ops.eltwise_hypot,
        "pytorch_op": pytorch_ops.hypot,
    },
    "ttnn-eltwise-i0": {
        "tt_lib_op": ttnn_ops.eltwise_i0,
        "pytorch_op": pytorch_ops.i0,
    },
    "ttnn-eltwise-isfinite": {
        "tt_lib_op": ttnn_ops.eltwise_isfinite,
        "pytorch_op": pytorch_ops.isfinite,
    },
    "ttnn-eltwise-isinf": {
        "tt_lib_op": ttnn_ops.eltwise_isinf,
        "pytorch_op": pytorch_ops.isinf,
    },
    "ttnn-eltwise-isnan": {
        "tt_lib_op": ttnn_ops.eltwise_isnan,
        "pytorch_op": pytorch_ops.isnan,
    },
    "ttnn-eltwise-isneginf": {
        "tt_lib_op": ttnn_ops.eltwise_isneginf,
        "pytorch_op": pytorch_ops.isneginf,
    },
    "ttnn-eltwise-isposinf": {
        "tt_lib_op": ttnn_ops.eltwise_isposinf,
        "pytorch_op": pytorch_ops.isposinf,
    },
    "ttnn-eltwise-leaky_relu": {
        "tt_lib_op": ttnn_ops.eltwise_leaky_relu,
        "pytorch_op": pytorch_ops.leaky_relu,
    },
    "ttnn-eltwise-lgamma": {
        "tt_lib_op": ttnn_ops.eltwise_lgamma,
        "pytorch_op": pytorch_ops.lgamma,
    },
    "ttnn-eltwise-log": {
        "tt_lib_op": ttnn_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "ttnn-eltwise-log10": {
        "tt_lib_op": ttnn_ops.eltwise_log10,
        "pytorch_op": pytorch_ops.log10,
    },
    "ttnn-eltwise-log1p": {
        "tt_lib_op": ttnn_ops.eltwise_log1p,
        "pytorch_op": pytorch_ops.log1p,
    },
    "ttnn-eltwise-log2": {
        "tt_lib_op": ttnn_ops.eltwise_log2,
        "pytorch_op": pytorch_ops.log2,
    },
    "ttnn-eltwise-log_sigmoid": {
        "tt_lib_op": ttnn_ops.eltwise_log_sigmoid,
        "pytorch_op": pytorch_ops.log_sigmoid,
    },
    "ttnn-eltwise-logit": {
        "tt_lib_op": ttnn_ops.eltwise_logit,
        "pytorch_op": pytorch_ops.logit,
    },
    "ttnn-eltwise-mish": {
        "tt_lib_op": ttnn_ops.eltwise_mish,
        "pytorch_op": pytorch_ops.mish,
    },
    "ttnn-eltwise-multigammaln": {
        "tt_lib_op": ttnn_ops.eltwise_multigammaln,
        "pytorch_op": pytorch_ops.multigammaln,
    },
    "ttnn-eltwise-neg": {
        "tt_lib_op": ttnn_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
    },
    "ttnn-eltwise-prelu": {
        "tt_lib_op": ttnn_ops.eltwise_prelu,
        "pytorch_op": pytorch_ops.ttnn_prelu,
    },
    "ttnn-eltwise-relu": {
        "tt_lib_op": ttnn_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "ttnn-eltwise-logical_not": {
        "tt_lib_op": ttnn_ops.eltwise_logical_not,
        "pytorch_op": pytorch_ops.logical_not,
    },
    "ttnn-eltwise-xlogy": {
        "tt_lib_op": ttnn_ops.eltwise_xlogy,
        "pytorch_op": pytorch_ops.xlogy,
    },
    "ttnn-eltwise-squared_difference": {
        "tt_lib_op": ttnn_ops.eltwise_squared_difference,
        "pytorch_op": pytorch_ops.squared_difference,
    },
    "ttnn-eltwise-add_and_apply_activation": {
        "tt_lib_op": ttnn_ops.eltwise_add_and_apply_activation,
        "pytorch_op": pytorch_ops.add_and_apply_activation,
    },
    "ttnn-eltwise-add_and_apply_activation_": {
        "tt_lib_op": ttnn_ops.eltwise_add_and_apply_activation_,
        "pytorch_op": pytorch_ops.add_and_apply_activation,
    },
    "ttnn-eltwise-gtz": {
        "tt_lib_op": ttnn_ops.eltwise_gtz,
        "pytorch_op": pytorch_ops.gtz,
    },
    "ttnn-eltwise-ltz": {
        "tt_lib_op": ttnn_ops.eltwise_ltz,
        "pytorch_op": pytorch_ops.ltz,
    },
    "ttnn-eltwise-gez": {
        "tt_lib_op": ttnn_ops.eltwise_gez,
        "pytorch_op": pytorch_ops.gez,
    },
    "ttnn-eltwise-lez": {
        "tt_lib_op": ttnn_ops.eltwise_lez,
        "pytorch_op": pytorch_ops.lez,
    },
    "ttnn-eltwise-nez": {
        "tt_lib_op": ttnn_ops.eltwise_nez,
        "pytorch_op": pytorch_ops.nez,
    },
    "ttnn-eltwise-add": {
        "tt_lib_op": ttnn_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "ttnn-eltwise-exp": {
        "tt_lib_op": ttnn_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "ttnn-permute": {
        "tt_lib_op": ttnn_ops.permute,
        "pytorch_op": pytorch_ops.permute,
    },
    "ttnn-reshape": {
        "tt_lib_op": ttnn_ops.reshape,
        "pytorch_op": pytorch_ops.reshape,
    },
    "ttnn-gelu": {
        "tt_lib_op": ttnn_ops.gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "ttnn-eltwise-sub": {
        "tt_lib_op": ttnn_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "ttnn-embeddings": {
        "tt_lib_op": ttnn_ops.embeddings,
        "pytorch_op": pytorch_ops.ttnn_embeddings,
    },
    "ttnn-eltwise-tanh": {
        "tt_lib_op": ttnn_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    "ttnn-eltwise-softmax": {
        "tt_lib_op": ttnn_ops.eltwise_softmax,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "ttnn-mul": {
        "tt_lib_op": ttnn_ops.mul,
        "pytorch_op": pytorch_ops.mul,
    },
    "ttnn-linear": {"tt_lib_op": ttnn_ops.linear, "pytorch_op": pytorch_ops.linear},
    "ttnn-eltwise-softmax_in_place": {
        "tt_lib_op": ttnn_ops.eltwise_softmax_in_place,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "ttnn-matmul": {
        "tt_lib_op": ttnn_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "ttnn-layernorm": {
        "tt_lib_op": ttnn_ops.layernorm,
        "pytorch_op": pytorch_ops.ttnn_layernorm_weights_bias,
    },
    "ttnn-layernorm_residual": {
        "tt_lib_op": ttnn_ops.layernorm_residual,
        "pytorch_op": pytorch_ops.ttnn_layernorm_weights_bias_residual,
    },
    "ttnn-layernorm_noweights": {
        "tt_lib_op": ttnn_ops.layernorm_noweights,
        "pytorch_op": pytorch_ops.ttnn_layernorm_noweights,
    },
    "abs-bw": {
        "tt_lib_op": tt_lib_ops.abs_bw,
        "pytorch_op": pytorch_ops.abs_bw,
    },
    "sqrt-bw": {
        "tt_lib_op": tt_lib_ops.sqrt_bw,
        "pytorch_op": pytorch_ops.sqrt_bw,
    },
    "relu-bw": {
        "tt_lib_op": tt_lib_ops.relu_bw,
        "pytorch_op": pytorch_ops.relu_bw,
    },
    "neg-bw": {
        "tt_lib_op": tt_lib_ops.relu_bw,
        "pytorch_op": pytorch_ops.relu_bw,
    },
    "log-bw": {
        "tt_lib_op": tt_lib_ops.log_bw,
        "pytorch_op": pytorch_ops.log_bw,
    },
    "gt-bw": {
        "tt_lib_op": tt_lib_ops.gt_bw,
        "pytorch_op": pytorch_ops.gt_bw,
    },
    "lt-bw": {
        "tt_lib_op": tt_lib_ops.gt_bw,
        "pytorch_op": pytorch_ops.gt_bw,
    },
    "ne-bw": {
        "tt_lib_op": tt_lib_ops.ne_bw,
        "pytorch_op": pytorch_ops.ne_bw,
    },
    "rsub-bw": {
        "tt_lib_op": tt_lib_ops.rsub_bw,
        "pytorch_op": pytorch_ops.rsub_bw,
    },
    "binary-le-bw": {
        "tt_lib_op": tt_lib_ops.binary_le_bw,
        "pytorch_op": pytorch_ops.binary_le_bw,
    },
    "clamp-max-bw": {
        "tt_lib_op": tt_lib_ops.clamp_max_bw,
        "pytorch_op": pytorch_ops.clamp_max_bw,
    },
    "clamp-min-bw": {
        "tt_lib_op": tt_lib_ops.clamp_min_bw,
        "pytorch_op": pytorch_ops.clamp_min_bw,
    },
    "clamp-bw": {
        "tt_lib_op": tt_lib_ops.clamp_bw,
        "pytorch_op": pytorch_ops.clamp_bw,
    },
    "ttnn-attention_softmax-nomask": {
        "tt_lib_op": ttnn_ops.attention_softmax_nomask,
        "pytorch_op": pytorch_ops.attention_softmax_nomask,
    },
    "ttnn-attention_softmax": {
        "tt_lib_op": ttnn_ops.attention_softmax,
        "pytorch_op": pytorch_ops.attention_softmax,
    },
    "ttnn-rmsnorm": {
        "tt_lib_op": ttnn_ops.rmsnorm,
        "pytorch_op": pytorch_ops.ttnn_rmsnorm,
    },
    "ttnn-transformer_concatenate_heads": {
        "tt_lib_op": ttnn_ops.transformer_concatenate_heads,
        "pytorch_op": pytorch_ops.transformer_concatenate_heads,
    },
    "ttnn-full-like": {
        "tt_lib_op": ttnn_ops.full_like,
        "pytorch_op": pytorch_ops.full_like,
    },
    "ttnn-abs": {
        "tt_lib_op": ttnn_ops.abs,
        "pytorch_op": pytorch_ops.abs,
    },
    "ttnn-acos": {
        "tt_lib_op": ttnn_ops.acos,
        "pytorch_op": pytorch_ops.acos,
    },
    "ttnn-acosh": {
        "tt_lib_op": ttnn_ops.acosh,
        "pytorch_op": pytorch_ops.acosh,
    },
    "ttnn-asin": {
        "tt_lib_op": ttnn_ops.asin,
        "pytorch_op": pytorch_ops.asin,
    },
    "ttnn-asinh": {
        "tt_lib_op": ttnn_ops.asinh,
        "pytorch_op": pytorch_ops.asinh,
    },
    "ttnn-atan": {
        "tt_lib_op": ttnn_ops.atan,
        "pytorch_op": pytorch_ops.atan,
    },
    "ttnn-atan2": {
        "tt_lib_op": ttnn_ops.atan2,
        "pytorch_op": pytorch_ops.atan2,
    },
    "ttnn-atanh": {
        "tt_lib_op": ttnn_ops.atanh,
        "pytorch_op": pytorch_ops.atanh,
    },
    "ttnn-cos": {
        "tt_lib_op": ttnn_ops.cos,
        "pytorch_op": pytorch_ops.cos,
    },
    "ttnn-cosh": {
        "tt_lib_op": ttnn_ops.cosh,
        "pytorch_op": pytorch_ops.cosh,
    },
    "ttnn-exp": {
        "tt_lib_op": ttnn_ops.exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "ttnn-exp2": {
        "tt_lib_op": ttnn_ops.exp2,
        "pytorch_op": pytorch_ops.exp2,
    },
    "ttnn-expm1": {
        "tt_lib_op": ttnn_ops.expm1,
        "pytorch_op": pytorch_ops.expm1,
    },
    "ttnn-erf": {
        "tt_lib_op": ttnn_ops.erf,
        "pytorch_op": pytorch_ops.erf,
    },
    "ttnn-erfc": {
        "tt_lib_op": ttnn_ops.erfc,
        "pytorch_op": pytorch_ops.erfc,
    },
    "ttnn-elu": {
        "tt_lib_op": ttnn_ops.elu,
        "pytorch_op": pytorch_ops.elu,
    },
    "ttnn-erfinv": {
        "tt_lib_op": ttnn_ops.erfinv,
        "pytorch_op": pytorch_ops.erfinv,
    },
    "ttnn-hardsigmoid": {
        "tt_lib_op": ttnn_ops.hardsigmoid,
        "pytorch_op": pytorch_ops.hardsigmoid,
    },
    "ttnn-deg2rad": {
        "tt_lib_op": ttnn_ops.deg2rad,
        "pytorch_op": pytorch_ops.deg2rad,
    },
    "ttnn-hardshrink": {
        "tt_lib_op": ttnn_ops.hardshrink,
        "pytorch_op": pytorch_ops.hardshrink,
    },
    "ttnn-cbrt": {
        "tt_lib_op": ttnn_ops.cbrt,
        "pytorch_op": pytorch_ops.cbrt,
    },
    "ttnn-clone": {
        "tt_lib_op": ttnn_ops.clone,
        "pytorch_op": pytorch_ops.clone,
    },
    "ttnn-digamma": {
        "tt_lib_op": ttnn_ops.digamma,
        "pytorch_op": pytorch_ops.digamma,
    },
    "ttnn-clip": {
        "tt_lib_op": ttnn_ops.clip,
        "pytorch_op": pytorch_ops.clip,
    },
    "ttnn-repeat_interleave": {
        "tt_lib_op": ttnn_ops.repeat_interleave,
        "pytorch_op": pytorch_ops.repeat_interleave,
    },
    "ttnn-addcmul": {
        "tt_lib_op": ttnn_ops.addcmul,
        "pytorch_op": pytorch_ops.addcmul,
    },
    "ttnn-groupnorm_noweights": {
        "tt_lib_op": ttnn_ops.groupnorm_noweights,
        "pytorch_op": pytorch_ops.groupnorm_noweights,
    },
    "ttnn-groupnorm": {
        "tt_lib_op": ttnn_ops.groupnorm,
        "pytorch_op": pytorch_ops.ttnn_groupnorm,
    },
    "ttnn-global-avg-pool2d": {
        "tt_lib_op": ttnn_ops.global_avg_pool2d,
        "pytorch_op": pytorch_ops.global_avg_pool2d,
    },
    "ttnn-upsample": {
        "tt_lib_op": ttnn_ops.upsample,
        "pytorch_op": pytorch_ops.upsample,
    },
    "ttnn-l1_loss": {
        "tt_lib_op": ttnn_ops.l1_loss,
        "pytorch_op": pytorch_ops.l1_loss,
    },
    "ttnn-l1_loss_sum": {
        "tt_lib_op": ttnn_ops.l1_loss_sum,
        "pytorch_op": pytorch_ops.l1_loss_sum,
    },
    "ttnn-l1_loss_mean": {
        "tt_lib_op": ttnn_ops.l1_loss_mean,
        "pytorch_op": pytorch_ops.l1_loss_mean,
    },
    "ttnn-mse_loss": {
        "tt_lib_op": ttnn_ops.mse_loss,
        "pytorch_op": pytorch_ops.mse_loss,
    },
    "ttnn-mse_loss_sum": {
        "tt_lib_op": ttnn_ops.mse_loss_sum,
        "pytorch_op": pytorch_ops.mse_loss_sum,
    },
    "ttnn-mse_loss_mean": {
        "tt_lib_op": ttnn_ops.mse_loss_mean,
        "pytorch_op": pytorch_ops.mse_loss_mean,
    },
    "ttnn-ldexp": {
        "tt_lib_op": ttnn_ops.ldexp,
        "pytorch_op": pytorch_ops.ldexp,
    },
    "ttnn-logical_xor": {
        "tt_lib_op": ttnn_ops.logical_xor,
        "pytorch_op": pytorch_ops.logical_xor,
    },
    "ttnn-logical_and": {
        "tt_lib_op": ttnn_ops.logical_and,
        "pytorch_op": pytorch_ops.logical_and,
    },
    "ttnn-logical_or": {
        "tt_lib_op": ttnn_ops.logical_or,
        "pytorch_op": pytorch_ops.logical_or,
    },
    "ttnn-pow": {
        "tt_lib_op": ttnn_ops.pow,
        "pytorch_op": pytorch_ops.power,
    },
    "ttnn-logaddexp2": {
        "tt_lib_op": ttnn_ops.logaddexp2,
        "pytorch_op": pytorch_ops.logaddexp2,
    },
    "ttnn-logaddexp": {
        "tt_lib_op": ttnn_ops.logaddexp,
        "pytorch_op": pytorch_ops.logaddexp,
    },
}
