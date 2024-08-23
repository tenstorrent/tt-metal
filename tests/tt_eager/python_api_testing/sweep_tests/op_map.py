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
    "fill-rm": {
        "tt_op": tt_lib_ops.fill_rm,
        "pytorch_op": pytorch_ops.fill_rm,
    },
    "fill-ones-rm": {
        "tt_op": tt_lib_ops.fill_ones_rm,
        "pytorch_op": pytorch_ops.fill_ones_rm,
    },
    "fill-zero-bw": {
        "tt_op": tt_lib_ops.fill_zero_bw,
        "pytorch_op": pytorch_ops.fill_zero_bw,
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
    "eltwise-bitwise_complement": {
        "tt_op": None,  # tt_lib_ops.eltwise_bitwise_complement,
        "pytorch_op": None,  # pytorch_ops.bitwise_complement,
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
    "eltwise-elu": {
        "tt_op": tt_lib_ops.eltwise_elu,
        "pytorch_op": pytorch_ops.elu,
    },
    "eltwise-div": {
        "tt_op": tt_lib_ops.eltwise_div,
        "pytorch_op": pytorch_ops.div,
    },
    "eltwise-unary_rdiv_trunc": {
        "tt_op": tt_lib_ops.eltwise_unary_rdiv_trunc,
        "pytorch_op": pytorch_ops.unary_rdiv_trunc,
    },
    "eltwise-square": {
        "tt_op": tt_lib_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "eltwise-softplus": {
        "tt_op": tt_lib_ops.eltwise_softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "eltwise-neg": {
        "tt_op": tt_lib_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
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
    "eltwise-acos": {
        "tt_op": tt_lib_ops.eltwise_acos,
        "pytorch_op": pytorch_ops.acos,
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
    "eltwise-rsqrt": {
        "tt_op": tt_lib_ops.eltwise_rsqrt,
        "pytorch_op": pytorch_ops.rsqrt,
    },
    "eltwise-logical_and": {
        "tt_op": tt_lib_ops.eltwise_logical_and,
        "pytorch_op": pytorch_ops.logical_and,
    },
    "eltwise-leaky_relu": {
        "tt_op": tt_lib_ops.eltwise_leaky_relu,
        "pytorch_op": pytorch_ops.leaky_relu,
    },
    "eltwise-prelu": {
        "tt_op": tt_lib_ops.eltwise_prelu,
        "pytorch_op": pytorch_ops.prelu,
    },
    "eltwise-bias_gelu_unary": {
        "tt_op": tt_lib_ops.eltwise_bias_gelu_unary,
        "pytorch_op": pytorch_ops.bias_gelu_unary,
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
    "eltwise-celu": {
        "tt_op": tt_lib_ops.eltwise_celu,
        "pytorch_op": pytorch_ops.celu,
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
    "eltwise-right_shift": {
        "tt_op": tt_lib_ops.eltwise_right_shift,
        "pytorch_op": pytorch_ops.right_shift,
    },
    "eltwise-left_shift": {
        "tt_op": tt_lib_ops.eltwise_left_shift,
        "pytorch_op": pytorch_ops.left_shift,
    },
    "eltwise-fmod": {
        "tt_op": tt_lib_ops.eltwise_fmod,
        "pytorch_op": pytorch_ops.fmod,
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
    "eltwise-signbit": {
        "tt_op": tt_lib_ops.eltwise_signbit,
        "pytorch_op": pytorch_ops.signbit,
    },
    "eltwise-floor_div": {
        "tt_op": tt_lib_ops.eltwise_floor_div,
        "pytorch_op": pytorch_ops.floor_div,
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
    "eltwise-scatter": {
        "tt_op": tt_lib_ops.eltwise_scatter,
        "pytorch_op": pytorch_ops.scatter,
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
    "eltwise-logical_or": {
        "tt_op": tt_lib_ops.eltwise_logical_or,
        "pytorch_op": pytorch_ops.logical_or,
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
    "transpose": {
        "tt_op": tt_lib_ops.transpose,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
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
    "complex-div": {
        "tt_op": tt_lib_ops.complex_div,
        "pytorch_op": pytorch_ops.complex_div,
    },
    "complex-mul": {
        "tt_op": tt_lib_ops.complex_mul,
        "pytorch_op": pytorch_ops.complex_mul,
    },
    "complex-imag": {
        "tt_op": tt_lib_ops.complex_imag,
        "pytorch_op": pytorch_ops.complex_imag,
    },
    "unary-div-bw": {
        "tt_op": tt_lib_ops.unary_div_bw,
        "pytorch_op": pytorch_ops.unary_div_bw,
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
    "abs-bw": {
        "tt_op": tt_lib_ops.abs_bw,
        "pytorch_op": pytorch_ops.abs_bw,
    },
    "binary-le-bw": {
        "tt_op": tt_lib_ops.binary_le_bw,
        "pytorch_op": pytorch_ops.binary_le_bw,
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
