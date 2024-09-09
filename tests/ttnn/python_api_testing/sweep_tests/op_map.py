# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.ttnn.python_api_testing.sweep_tests import ttnn_pytorch_ops
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


op_map = {
    "ttnn-concat": {
        "tt_op": ttnn_ops.concat,
        "pytorch_op": pytorch_ops.concat,
    },
    "ttnn-stats-var_global": {
        "tt_op": ttnn_ops.var,
        "pytorch_op": ttnn_pytorch_ops.var_global,
    },
    "ttnn-stats-std_global": {
        "tt_op": ttnn_ops.std,
        "pytorch_op": ttnn_pytorch_ops.std_global,
    },
    "ttnn-stats-mean_global": {
        "tt_op": ttnn_ops.mean,
        "pytorch_op": ttnn_pytorch_ops.mean_global,
    },
    "ttnn-tril": {
        "tt_op": ttnn_ops.tril,
        "pytorch_op": pytorch_ops.tril,
    },
    "ttnn-triu": {
        "tt_op": ttnn_ops.triu,
        "pytorch_op": pytorch_ops.triu,
    },
    "eltwise-sinh": {
        "tt_op": ttnn_ops.eltwise_sinh,
        "pytorch_op": pytorch_ops.sinh,
    },
    "ttnn-eltwise-eqz": {
        "tt_op": ttnn_ops.eltwise_eqz,
        "pytorch_op": pytorch_ops.eqz,
    },
    "ttnn-eltwise-sign": {
        "tt_op": ttnn_ops.eltwise_sign,
        "pytorch_op": pytorch_ops.sign,
    },
    "eltwise-silu": {
        "tt_op": ttnn_ops.eltwise_silu,
        "pytorch_op": pytorch_ops.silu,
    },
    "ttnn-eltwise-square": {
        "tt_op": ttnn_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "ttnn-softplus": {
        "tt_op": ttnn_ops.softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "eltwise-swish": {
        "tt_op": ttnn_ops.eltwise_swish,
        "pytorch_op": pytorch_ops.swish,
    },
    "ttnn-eltwise-sin": {
        "tt_op": ttnn_ops.eltwise_sin,
        "pytorch_op": pytorch_ops.sin,
    },
    "ttnn-eltwise-tan": {
        "tt_op": ttnn_ops.eltwise_tan,
        "pytorch_op": pytorch_ops.tan,
    },
    "ttnn-eltwise-recip": {
        "tt_op": ttnn_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "ttnn-eltwise-sqrt": {
        "tt_op": ttnn_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "ttnn-rsqrt": {
        "tt_op": ttnn_ops.eltwise_rsqrt,
        "pytorch_op": pytorch_ops.rsqrt,
    },
    "eltwise-lerp_binary": {
        "tt_op": ttnn_ops.eltwise_lerp_binary,
        "pytorch_op": pytorch_ops.lerp_binary,
    },
    "eltwise-addalpha": {
        "tt_op": ttnn_ops.eltwise_addalpha,
        "pytorch_op": pytorch_ops.addalpha,
    },
    "eltwise-lerp_ternary": {
        "tt_op": ttnn_ops.eltwise_lerp_ternary,
        "pytorch_op": pytorch_ops.lerp_ternary,
    },
    "eltwise-softshrink": {
        "tt_op": ttnn_ops.eltwise_softshrink,
        "pytorch_op": pytorch_ops.softshrink,
    },
    "ttnn-eltwise-softsign": {
        "tt_op": ttnn_ops.eltwise_softsign,
        "pytorch_op": pytorch_ops.softsign,
    },
    "eltwise-polyval": {
        "tt_op": ttnn_ops.eltwise_polyval,
        "pytorch_op": pytorch_ops.polyval,
    },
    "eltwise-mac": {
        "tt_op": ttnn_ops.eltwise_mac,
        "pytorch_op": pytorch_ops.mac,
    },
    "eltwise-addcdiv": {
        "tt_op": ttnn_ops.eltwise_addcdiv,
        "pytorch_op": pytorch_ops.addcdiv,
    },
    "ttnn-eltwise-sigmoid": {
        "tt_op": ttnn_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "ttnn-eltwise-sigmoid_accurate": {
        "tt_op": ttnn_ops.eltwise_sigmoid_accurate,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-polygamma": {
        "tt_op": ttnn_ops.eltwise_polygamma,
        "pytorch_op": pytorch_ops.polygamma,
    },
    "eltwise-tanhshrink": {
        "tt_op": ttnn_ops.eltwise_tanhshrink,
        "pytorch_op": pytorch_ops.tanhshrink,
    },
    "ttnn-eltwise-signbit": {
        "tt_op": ttnn_ops.eltwise_signbit,
        "pytorch_op": pytorch_ops.signbit,
    },
    "ttnn-eltwise-ne": {
        "tt_op": ttnn_ops.eltwise_ne,
        "pytorch_op": pytorch_ops.ne,
    },
    "ttnn-eltwise-eq": {
        "tt_op": ttnn_ops.eltwise_eq,
        "pytorch_op": pytorch_ops.eq,
    },
    "ttnn-eltwise-lt": {
        "tt_op": ttnn_ops.eltwise_lt,
        "pytorch_op": pytorch_ops.lt,
    },
    "ttnn-eltwise-gt": {
        "tt_op": ttnn_ops.eltwise_gt,
        "pytorch_op": pytorch_ops.gt,
    },
    "ttnn-eltwise-gte": {
        "tt_op": ttnn_ops.eltwise_gte,
        "pytorch_op": pytorch_ops.gte,
    },
    "ttnn-eltwise-lte": {
        "tt_op": ttnn_ops.eltwise_lte,
        "pytorch_op": pytorch_ops.lte,
    },
    "ttnn-min": {
        "tt_op": ttnn_ops.min,
        "pytorch_op": ttnn_pytorch_ops.min,
    },
    "ttnn-max": {
        "tt_op": ttnn_ops.max,
        "pytorch_op": ttnn_pytorch_ops.max,
    },
    "ttnn-eltwise-min": {
        "tt_op": ttnn_ops.eltwise_min,
        "pytorch_op": ttnn_pytorch_ops.eltwise_min,
    },
    "ttnn-eltwise-max": {
        "tt_op": ttnn_ops.eltwise_max,
        "pytorch_op": ttnn_pytorch_ops.eltwise_max,
    },
    "ttnn-eltwise-rad2deg": {
        "tt_op": ttnn_ops.eltwise_rad2deg,
        "pytorch_op": pytorch_ops.rad2deg,
    },
    "eltwise-threshold": {
        "tt_op": ttnn_ops.eltwise_threshold,
        "pytorch_op": pytorch_ops.threshold,
    },
    "ttnn-relu6": {
        "tt_op": ttnn_ops.eltwise_relu6,
        "pytorch_op": pytorch_ops.relu6,
    },
    "eltwise-isclose": {
        "tt_op": ttnn_ops.eltwise_isclose,
        "pytorch_op": pytorch_ops.isclose,
    },
    "ttnn-eltwise-where": {
        "tt_op": ttnn_ops.where,
        "pytorch_op": pytorch_ops.where,
    },
    "ttnn-sum": {
        "tt_op": ttnn_ops.sum,
        "pytorch_op": pytorch_ops.sum,
    },
    "ttnn-activation_glu": {
        "tt_op": ttnn_ops.activation_glu,
        "pytorch_op": pytorch_ops.activation_glu,
    },
    "ttnn-activation_geglu": {
        "tt_op": ttnn_ops.activation_geglu,
        "pytorch_op": pytorch_ops.activation_geglu,
    },
    "ttnn-activation_swiglu": {
        "tt_op": ttnn_ops.activation_swiglu,
        "pytorch_op": pytorch_ops.activation_swiglu,
    },
    "eltwise-ones": {
        "tt_op": ttnn_ops.ones,
        "pytorch_op": pytorch_ops.ones,
    },
    "eltwise-ones_like": {
        "tt_op": ttnn_ops.ones_like,
        "pytorch_op": pytorch_ops.ones_like,
    },
    "eltwise-full": {
        "tt_op": ttnn_ops.full,
        "pytorch_op": pytorch_ops.full,
    },
    "eltwise-hardswish": {
        "tt_op": ttnn_ops.eltwise_hardswish,
        "pytorch_op": pytorch_ops.hardswish,
    },
    "ttnn-eltwise-hardtanh": {
        "tt_op": ttnn_ops.eltwise_hardtanh,
        "pytorch_op": pytorch_ops.hardtanh,
    },
    "ttnn-eltwise-heaviside": {
        "tt_op": ttnn_ops.eltwise_heaviside,
        "pytorch_op": pytorch_ops.heaviside,
    },
    "eltwise-hypot": {
        "tt_op": ttnn_ops.eltwise_hypot,
        "pytorch_op": pytorch_ops.hypot,
    },
    "ttnn-eltwise-i0": {
        "tt_op": ttnn_ops.eltwise_i0,
        "pytorch_op": pytorch_ops.i0,
    },
    "ttnn-eltwise-isfinite": {
        "tt_op": ttnn_ops.eltwise_isfinite,
        "pytorch_op": pytorch_ops.isfinite,
    },
    "ttnn-eltwise-isinf": {
        "tt_op": ttnn_ops.eltwise_isinf,
        "pytorch_op": pytorch_ops.isinf,
    },
    "ttnn-eltwise-isnan": {
        "tt_op": ttnn_ops.eltwise_isnan,
        "pytorch_op": pytorch_ops.isnan,
    },
    "ttnn-eltwise-isneginf": {
        "tt_op": ttnn_ops.eltwise_isneginf,
        "pytorch_op": pytorch_ops.isneginf,
    },
    "ttnn-eltwise-isposinf": {
        "tt_op": ttnn_ops.eltwise_isposinf,
        "pytorch_op": pytorch_ops.isposinf,
    },
    "ttnn-eltwise-leaky_relu": {
        "tt_op": ttnn_ops.eltwise_leaky_relu,
        "pytorch_op": pytorch_ops.leaky_relu,
    },
    "eltwise-lgamma": {
        "tt_op": ttnn_ops.eltwise_lgamma,
        "pytorch_op": pytorch_ops.lgamma,
    },
    "ttnn-eltwise-log": {
        "tt_op": ttnn_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "ttnn-eltwise-log10": {
        "tt_op": ttnn_ops.eltwise_log10,
        "pytorch_op": pytorch_ops.log10,
    },
    "eltwise-log1p": {
        "tt_op": ttnn_ops.eltwise_log1p,
        "pytorch_op": pytorch_ops.log1p,
    },
    "ttnn-eltwise-log2": {
        "tt_op": ttnn_ops.eltwise_log2,
        "pytorch_op": pytorch_ops.log2,
    },
    "ttnn-eltwise-log_sigmoid": {
        "tt_op": ttnn_ops.eltwise_log_sigmoid,
        "pytorch_op": pytorch_ops.log_sigmoid,
    },
    "eltwise-logit": {
        "tt_op": ttnn_ops.eltwise_logit,
        "pytorch_op": pytorch_ops.logit,
    },
    "eltwise-mish": {
        "tt_op": ttnn_ops.eltwise_mish,
        "pytorch_op": pytorch_ops.mish,
    },
    "eltwise-multigammaln": {
        "tt_op": ttnn_ops.eltwise_multigammaln,
        "pytorch_op": pytorch_ops.multigammaln,
    },
    "ttnn-eltwise-neg": {
        "tt_op": ttnn_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
    },
    "ttnn-eltwise-prelu": {
        "tt_op": ttnn_ops.eltwise_prelu,
        "pytorch_op": ttnn_pytorch_ops.prelu,
    },
    "ttnn-eltwise-relu": {
        "tt_op": ttnn_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "ttnn-eltwise-logical_not": {
        "tt_op": ttnn_ops.eltwise_logical_not,
        "pytorch_op": pytorch_ops.logical_not,
    },
    "eltwise-xlogy": {
        "tt_op": ttnn_ops.eltwise_xlogy,
        "pytorch_op": pytorch_ops.xlogy,
    },
    "ttnn-eltwise-squared_difference": {
        "tt_op": ttnn_ops.eltwise_squared_difference,
        "pytorch_op": pytorch_ops.squared_difference,
    },
    "ttnn-eltwise-add_and_apply_activation": {
        "tt_op": ttnn_ops.eltwise_add_and_apply_activation,
        "pytorch_op": pytorch_ops.add_and_apply_activation,
    },
    "ttnn-eltwise-add_and_apply_activation_": {
        "tt_op": ttnn_ops.eltwise_add_and_apply_activation_,
        "pytorch_op": pytorch_ops.add_and_apply_activation,
    },
    "ttnn-eltwise-gtz": {
        "tt_op": ttnn_ops.eltwise_gtz,
        "pytorch_op": pytorch_ops.gtz,
    },
    "ttnn-eltwise-ltz": {
        "tt_op": ttnn_ops.eltwise_ltz,
        "pytorch_op": pytorch_ops.ltz,
    },
    "ttnn-eltwise-gez": {
        "tt_op": ttnn_ops.eltwise_gez,
        "pytorch_op": pytorch_ops.gez,
    },
    "ttnn-eltwise-lez": {
        "tt_op": ttnn_ops.eltwise_lez,
        "pytorch_op": pytorch_ops.lez,
    },
    "ttnn-eltwise-nez": {
        "tt_op": ttnn_ops.eltwise_nez,
        "pytorch_op": pytorch_ops.nez,
    },
    "ttnn-eltwise-add": {
        "tt_op": ttnn_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "ttnn-eltwise-exp": {
        "tt_op": ttnn_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "ttnn-permute": {
        "tt_op": ttnn_ops.permute,
        "pytorch_op": pytorch_ops.permute,
    },
    "ttnn-reshape": {
        "tt_op": ttnn_ops.reshape,
        "pytorch_op": pytorch_ops.reshape,
    },
    "ttnn-transpose_01": {
        "tt_op": ttnn_ops.transpose_01,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=1),
    },
    "ttnn-transpose_02": {
        "tt_op": ttnn_ops.transpose_02,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=2),
    },
    "ttnn-transpose_03": {
        "tt_op": ttnn_ops.transpose_03,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=3),
    },
    "ttnn-transpose_12": {
        "tt_op": ttnn_ops.transpose_12,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=1, dim1=2),
    },
    "ttnn-transpose_13": {
        "tt_op": ttnn_ops.transpose_13,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=3),
    },
    "ttnn-transpose_23": {
        "tt_op": ttnn_ops.transpose_23,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=2, dim1=3),
    },
    "ttnn-gelu": {
        "tt_op": ttnn_ops.gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "ttnn-eltwise-sub": {
        "tt_op": ttnn_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "ttnn-embeddings": {
        "tt_op": ttnn_ops.embeddings,
        "pytorch_op": ttnn_pytorch_ops.embeddings,
    },
    "ttnn-eltwise-tanh": {
        "tt_op": ttnn_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    "ttnn-softmax": {
        "tt_op": ttnn_ops.softmax,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "ttnn-mul": {
        "tt_op": ttnn_ops.mul,
        "pytorch_op": pytorch_ops.mul,
    },
    "ttnn-linear": {"tt_op": ttnn_ops.linear, "pytorch_op": pytorch_ops.linear},
    "ttnn-eltwise-softmax_in_place": {
        "tt_op": ttnn_ops.eltwise_softmax_in_place,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "ttnn-matmul": {
        "tt_op": ttnn_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "ttnn-layernorm": {
        "tt_op": ttnn_ops.layernorm,
        "pytorch_op": ttnn_pytorch_ops.layernorm_weights_bias,
    },
    "ttnn-layernorm_residual": {
        "tt_op": ttnn_ops.layernorm_residual,
        "pytorch_op": ttnn_pytorch_ops.layernorm_weights_bias_residual,
    },
    "ttnn-layernorm_noweights": {
        "tt_op": ttnn_ops.layernorm_noweights,
        "pytorch_op": ttnn_pytorch_ops.layernorm_noweights,
    },
    "ttnn-attention_softmax-nomask": {
        "tt_op": ttnn_ops.attention_softmax_nomask,
        "pytorch_op": ttnn_pytorch_ops.attention_softmax_nomask,
    },
    "ttnn-attention_softmax": {
        "tt_op": ttnn_ops.attention_softmax,
        "pytorch_op": ttnn_pytorch_ops.attention_softmax,
    },
    "ttnn-rmsnorm": {
        "tt_op": ttnn_ops.rmsnorm,
        "pytorch_op": ttnn_pytorch_ops.rmsnorm,
    },
    "ttnn-transformer_concatenate_heads": {
        "tt_op": ttnn_ops.transformer_concatenate_heads,
        "pytorch_op": ttnn_pytorch_ops.transformer_concatenate_heads,
    },
    "eltwise-full_like": {
        "tt_op": ttnn_ops.full_like,
        "pytorch_op": pytorch_ops.full_like,
    },
    "ttnn-abs": {
        "tt_op": ttnn_ops.abs,
        "pytorch_op": pytorch_ops.abs,
    },
    "ttnn-acos": {
        "tt_op": ttnn_ops.acos,
        "pytorch_op": pytorch_ops.acos,
    },
    "eltwise-acosh": {
        "tt_op": ttnn_ops.acosh,
        "pytorch_op": pytorch_ops.acosh,
    },
    "ttnn-asin": {
        "tt_op": ttnn_ops.asin,
        "pytorch_op": pytorch_ops.asin,
    },
    "eltwise-asinh": {
        "tt_op": ttnn_ops.asinh,
        "pytorch_op": pytorch_ops.asinh,
    },
    "ttnn-atan": {
        "tt_op": ttnn_ops.atan,
        "pytorch_op": pytorch_ops.atan,
    },
    "eltwise-atan2": {
        "tt_op": ttnn_ops.atan2,
        "pytorch_op": pytorch_ops.atan2,
    },
    "eltwise-atanh": {
        "tt_op": ttnn_ops.atanh,
        "pytorch_op": pytorch_ops.atanh,
    },
    "ttnn-cos": {
        "tt_op": ttnn_ops.cos,
        "pytorch_op": pytorch_ops.cos,
    },
    "eltwise-cosh": {
        "tt_op": ttnn_ops.cosh,
        "pytorch_op": pytorch_ops.cosh,
    },
    "ttnn-exp": {
        "tt_op": ttnn_ops.exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "ttnn-exp2": {
        "tt_op": ttnn_ops.exp2,
        "pytorch_op": pytorch_ops.exp2,
    },
    "ttnn-expm1": {
        "tt_op": ttnn_ops.expm1,
        "pytorch_op": pytorch_ops.expm1,
    },
    "ttnn-erf": {
        "tt_op": ttnn_ops.erf,
        "pytorch_op": pytorch_ops.erf,
    },
    "ttnn-erfc": {
        "tt_op": ttnn_ops.erfc,
        "pytorch_op": pytorch_ops.erfc,
    },
    "ttnn-elu": {
        "tt_op": ttnn_ops.elu,
        "pytorch_op": pytorch_ops.elu,
    },
    "ttnn-erfinv": {
        "tt_op": ttnn_ops.erfinv,
        "pytorch_op": pytorch_ops.erfinv,
    },
    "eltwise-hardsigmoid": {
        "tt_op": ttnn_ops.hardsigmoid,
        "pytorch_op": pytorch_ops.hardsigmoid,
    },
    "ttnn-deg2rad": {
        "tt_op": ttnn_ops.deg2rad,
        "pytorch_op": pytorch_ops.deg2rad,
    },
    "eltwise-hardshrink": {
        "tt_op": ttnn_ops.hardshrink,
        "pytorch_op": pytorch_ops.hardshrink,
    },
    "eltwise-cbrt": {
        "tt_op": ttnn_ops.cbrt,
        "pytorch_op": pytorch_ops.cbrt,
    },
    "ttnn-clone": {
        "tt_op": ttnn_ops.clone,
        "pytorch_op": pytorch_ops.clone,
    },
    "eltwise-digamma": {
        "tt_op": ttnn_ops.digamma,
        "pytorch_op": pytorch_ops.digamma,
    },
    "ttnn-clip": {
        "tt_op": ttnn_ops.clip,
        "pytorch_op": pytorch_ops.clip,
    },
    "ttnn-repeat_interleave": {
        "tt_op": ttnn_ops.repeat_interleave,
        "pytorch_op": pytorch_ops.repeat_interleave,
    },
    "eltwise-addcmul": {
        "tt_op": ttnn_ops.addcmul,
        "pytorch_op": pytorch_ops.addcmul,
    },
    "ttnn-groupnorm_noweights": {
        "tt_op": ttnn_ops.groupnorm_noweights,
        "pytorch_op": pytorch_ops.groupnorm_noweights,
    },
    "ttnn-global-avg-pool2d": {
        "tt_op": ttnn_ops.global_avg_pool2d,
        "pytorch_op": pytorch_ops.global_avg_pool2d,
    },
    "ttnn-max-pool2d": {
        "tt_op": ttnn_ops.max_pool2d_tt,
        "pytorch_op": pytorch_ops.max_pool2d,
    },
    "ttnn-upsample": {
        "tt_op": ttnn_ops.upsample,
        "pytorch_op": pytorch_ops.upsample,
    },
    "ttnn-l1_loss": {
        "tt_op": ttnn_ops.l1_loss,
        "pytorch_op": pytorch_ops.l1_loss,
    },
    "ttnn-l1_loss_sum": {
        "tt_op": ttnn_ops.l1_loss_sum,
        "pytorch_op": pytorch_ops.l1_loss_sum,
    },
    "ttnn-l1_loss_mean": {
        "tt_op": ttnn_ops.l1_loss_mean,
        "pytorch_op": pytorch_ops.l1_loss_mean,
    },
    "ttnn-mse_loss": {
        "tt_op": ttnn_ops.mse_loss,
        "pytorch_op": pytorch_ops.mse_loss,
    },
    "ttnn-mse_loss_sum": {
        "tt_op": ttnn_ops.mse_loss_sum,
        "pytorch_op": pytorch_ops.mse_loss_sum,
    },
    "ttnn-mse_loss_mean": {
        "tt_op": ttnn_ops.mse_loss_mean,
        "pytorch_op": pytorch_ops.mse_loss_mean,
    },
    "ttnn-ldexp": {
        "tt_op": ttnn_ops.ldexp,
        "pytorch_op": pytorch_ops.ldexp,
    },
    "eltwise-logical_xor": {
        "tt_op": ttnn_ops.logical_xor,
        "pytorch_op": pytorch_ops.logical_xor,
    },
    "ttnn-logical_and": {
        "tt_op": ttnn_ops.logical_and,
        "pytorch_op": pytorch_ops.logical_and,
    },
    "ttnn-logical_or": {
        "tt_op": ttnn_ops.logical_or,
        "pytorch_op": pytorch_ops.logical_or,
    },
    "ttnn-pow": {
        "tt_op": ttnn_ops.pow,
        "pytorch_op": pytorch_ops.power_2,
    },
    "ttnn-logaddexp2": {
        "tt_op": ttnn_ops.logaddexp2,
        "pytorch_op": pytorch_ops.logaddexp2,
    },
    "ttnn-logaddexp": {
        "tt_op": ttnn_ops.logaddexp,
        "pytorch_op": pytorch_ops.logaddexp,
    },
    "ttnn-rotary-embedding": {
        "tt_op": ttnn_ops.rotary_embedding,
        "pytorch_op": pytorch_ops.rotary_embedding,
    },
    "ttnn-activation_reglu": {
        "tt_op": ttnn_ops.activation_reglu,
        "pytorch_op": pytorch_ops.activation_reglu,
    },
    "eltwise-arange": {
        "tt_op": ttnn_ops.arange,
        "pytorch_op": pytorch_ops.arange,
    },
    "eltwise-nextafter": {
        "tt_op": ttnn_ops.nextafter,
        "pytorch_op": pytorch_ops.nextafter,
    },
    "eltwise-empty": {
        "tt_op": ttnn_ops.empty,
        "pytorch_op": pytorch_ops.empty,
    },
    "ttnn-attention_softmax_nomask_2": {
        "tt_op": ttnn_ops.attention_softmax_nomask_2,
        "pytorch_op": ttnn_pytorch_ops.attention_softmax_nomask,
    },
    "ttnn-attention_softmax_2": {
        "tt_op": ttnn_ops.attention_softmax_2,
        "pytorch_op": ttnn_pytorch_ops.attention_softmax,
    },
    "eltwise-zeros": {
        "tt_op": ttnn_ops.zeros,
        "pytorch_op": pytorch_ops.zeros,
    },
    "eltwise-zeros_like": {
        "tt_op": ttnn_ops.zeros_like,
        "pytorch_op": pytorch_ops.zeros_like,
    },
    "ttnn-preprocess-model-conv-conv": {
        "tt_op": ttnn_ops.preprocessing_model_conv_conv,
        "pytorch_op": ttnn_pytorch_ops.preprocessing_model_conv_conv,
    },
    "ttnn-preprocess-model-conv-relu-conv": {
        "tt_op": ttnn_ops.preprocessing_model_conv_relu_conv,
        "pytorch_op": ttnn_pytorch_ops.preprocessing_model_conv_relu_conv,
    },
    "ttnn-preprocess-model-bert-1": {
        "tt_op": ttnn_ops.preprocessing_model_bert_1,
        "pytorch_op": ttnn_pytorch_ops.preprocessing_model_bert_1,
    },
    "ttnn-preprocess-model-bert-2": {
        "tt_op": ttnn_ops.preprocessing_model_bert_2,
        "pytorch_op": ttnn_pytorch_ops.preprocessing_model_bert_2,
    },
    "ttnn-preprocess-model-bert-3": {
        "tt_op": ttnn_ops.preprocessing_model_bert_3,
        "pytorch_op": ttnn_pytorch_ops.preprocessing_model_bert_3,
    },
    "ttnn-preprocess-model-bert-4": {
        "tt_op": ttnn_ops.preprocessing_model_bert_4,
        "pytorch_op": ttnn_pytorch_ops.preprocessing_model_bert_4,
    },
    "ttnn-repeat_2": {
        "tt_op": ttnn_ops.repeat,
        "pytorch_op": pytorch_ops.repeat_2,
    },
    "ttnn-eltwise-subtract_and_apply_activation": {
        "tt_op": ttnn_ops.eltwise_subtract_and_apply_activation,
        "pytorch_op": pytorch_ops.subtract_and_apply_activation,
    },
    "ttnn-eltwise-subtract_and_apply_activation_": {
        "tt_op": ttnn_ops.eltwise_subtract_and_apply_activation_,
        "pytorch_op": pytorch_ops.subtract_and_apply_activation,
    },
    "ttnn-eltwise-multiply_and_apply_activation": {
        "tt_op": ttnn_ops.eltwise_multiply_and_apply_activation,
        "pytorch_op": pytorch_ops.multiply_and_apply_activation,
    },
    "ttnn-eltwise-multiply_and_apply_activation_": {
        "tt_op": ttnn_ops.eltwise_multiply_and_apply_activation_,
        "pytorch_op": pytorch_ops.multiply_and_apply_activation,
    },
    "pad": {
        "tt_op": ttnn_ops.pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "eltwise-relu_min": {
        "tt_op": ttnn_ops.eltwise_relu_min,
        "pytorch_op": pytorch_ops.relu_min,
    },
    "eltwise-relu_max": {
        "tt_op": ttnn_ops.eltwise_relu_max,
        "pytorch_op": pytorch_ops.relu_max,
    },
    "unpad": {
        "tt_op": ttnn_ops.unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    "eltwise-unary_fmod": {
        "tt_op": ttnn_ops.eltwise_unary_fmod,
        "pytorch_op": pytorch_ops.unary_fmod,
    },
    "eltwise-softmax_in_place": {
        "tt_op": ttnn_ops.eltwise_softmax_in_place,
        "pytorch_op": pytorch_ops.softmax_in_place,
    },
    "eltwise-scale_mask_softmax_in_place": {
        "tt_op": ttnn_ops.eltwise_scale_mask_softmax_in_place,
        "pytorch_op": pytorch_ops.scale_mask_softmax_in_place,
    },
    "unary-add-bw": {
        "tt_op": ttnn_ops.unary_add_bw,
        "pytorch_op": pytorch_ops.unary_add_bw,
    },
    "eltwise-add-bw": {
        "tt_op": ttnn_ops.eltwise_add_bw,
        "pytorch_op": pytorch_ops.add_bw,
    },
    "eltwise-rdiv": {
        "tt_op": ttnn_ops.eltwise_rdiv,
        "pytorch_op": pytorch_ops.eltwise_rdiv,
    },
    "eltwise-rsub": {
        "tt_op": ttnn_ops.eltwise_rsub,
        "pytorch_op": pytorch_ops.eltwise_rsub,
    },
    "log-bw": {
        "tt_op": ttnn_ops.log_bw,
        "pytorch_op": pytorch_ops.log_bw,
    },
    "neg-bw": {
        "tt_op": ttnn_ops.neg_bw,
        "pytorch_op": pytorch_ops.neg_bw,
    },
    "clamp-bw": {
        "tt_op": ttnn_ops.clamp_bw,
        "pytorch_op": pytorch_ops.clamp_bw,
    },
    "clamp-max-bw": {
        "tt_op": ttnn_ops.clamp_max_bw,
        "pytorch_op": pytorch_ops.clamp_max_bw,
    },
    "clamp-min-bw": {
        "tt_op": ttnn_ops.clamp_min_bw,
        "pytorch_op": pytorch_ops.clamp_min_bw,
    },
    "rsub-bw": {
        "tt_op": ttnn_ops.rsub_bw,
        "pytorch_op": pytorch_ops.rsub_bw,
    },
    "where-bw": {
        "tt_op": ttnn_ops.where_bw,
        "pytorch_op": pytorch_ops.where_bw,
    },
    "rsqrt-bw": {
        "tt_op": ttnn_ops.rsqrt_bw,
        "pytorch_op": pytorch_ops.rsqrt_bw,
    },
    "sqrt-bw": {
        "tt_op": ttnn_ops.sqrt_bw,
        "pytorch_op": pytorch_ops.sqrt_bw,
    },
    "addcmul-bw": {
        "tt_op": ttnn_ops.addcmul_bw,
        "pytorch_op": pytorch_ops.addcmul_bw,
    },
    "addcdiv-bw": {
        "tt_op": ttnn_ops.addcdiv_bw,
        "pytorch_op": pytorch_ops.addcdiv_bw,
    },
    "addalpha-bw": {
        "tt_op": ttnn_ops.addalpha_bw,
        "pytorch_op": pytorch_ops.addalpha_bw,
    },
    "subalpha-bw": {
        "tt_op": ttnn_ops.subalpha_bw,
        "pytorch_op": pytorch_ops.subalpha_bw,
    },
    "relu-bw": {
        "tt_op": ttnn_ops.relu_bw,
        "pytorch_op": pytorch_ops.relu_bw,
    },
    "gt-bw": {
        "tt_op": ttnn_ops.gt_bw,
        "pytorch_op": pytorch_ops.gt_bw,
    },
    "lt-bw": {
        "tt_op": ttnn_ops.gt_bw,
        "pytorch_op": pytorch_ops.gt_bw,
    },
    "ne-bw": {
        "tt_op": ttnn_ops.ne_bw,
        "pytorch_op": pytorch_ops.ne_bw,
    },
    "log10-bw": {
        "tt_op": ttnn_ops.log10_bw,
        "pytorch_op": pytorch_ops.log10_bw,
    },
    "log2-bw": {
        "tt_op": ttnn_ops.log2_bw,
        "pytorch_op": pytorch_ops.log2_bw,
    },
    "log1p-bw": {
        "tt_op": ttnn_ops.log1p_bw,
        "pytorch_op": pytorch_ops.log1p_bw,
    },
    "eltwise-subalpha": {
        "tt_op": ttnn_ops.eltwise_subalpha,
        "pytorch_op": pytorch_ops.subalpha,
    },
    "log-sigmoid-bw": {
        "tt_op": ttnn_ops.log_sigmoid_bw,
        "pytorch_op": pytorch_ops.log_sigmoid_bw,
    },
    "logaddexp-bw": {
        "tt_op": ttnn_ops.logaddexp_bw,
        "pytorch_op": pytorch_ops.logaddexp_bw,
    },
    "logaddexp2-bw": {
        "tt_op": ttnn_ops.logaddexp_bw,
        "pytorch_op": pytorch_ops.logaddexp_bw,
    },
    "fmod-bw": {
        "tt_op": ttnn_ops.fmod_bw,
        "pytorch_op": pytorch_ops.fmod_bw,
    },
    "frac-bw": {
        "tt_op": ttnn_ops.frac_bw,
        "pytorch_op": pytorch_ops.frac_bw,
    },
    "gelu-bw": {
        "tt_op": ttnn_ops.gelu_bw,
        "pytorch_op": pytorch_ops.gelu_bw,
    },
    "hardshrink-bw": {
        "tt_op": ttnn_ops.hardshrink_bw,
        "pytorch_op": pytorch_ops.hardshrink_bw,
    },
    "hardtanh-bw": {
        "tt_op": ttnn_ops.hardtanh_bw,
        "pytorch_op": pytorch_ops.hardtanh_bw,
    },
    "hypot-bw": {
        "tt_op": ttnn_ops.hypot_bw,
        "pytorch_op": pytorch_ops.hypot_bw,
    },
    "i0-bw": {
        "tt_op": ttnn_ops.i0_bw,
        "pytorch_op": pytorch_ops.i0_bw,
    },
    "ceil-bw": {
        "tt_op": ttnn_ops.ceil_bw,
        "pytorch_op": pytorch_ops.ceil_bw,
    },
    "celu-bw": {
        "tt_op": ttnn_ops.celu_bw,
        "pytorch_op": pytorch_ops.celu_bw,
    },
    "cosh-bw": {
        "tt_op": ttnn_ops.cosh_bw,
        "pytorch_op": pytorch_ops.cosh_bw,
    },
    "cos-bw": {
        "tt_op": ttnn_ops.cos_bw,
        "pytorch_op": pytorch_ops.cos_bw,
    },
    "stats-normalize_global": {
        "tt_op": ttnn_ops.normalize_global,
        "pytorch_op": pytorch_ops.normalize_global,
    },
    "stats-normalize_hw": {
        "tt_op": ttnn_ops.normalize_hw,
        "pytorch_op": pytorch_ops.normalize_hw,
    },
    "erf-bw": {
        "tt_op": ttnn_ops.erf_bw,
        "pytorch_op": pytorch_ops.erf_bw,
    },
    "erfc-bw": {
        "tt_op": ttnn_ops.erfc_bw,
        "pytorch_op": pytorch_ops.erfc_bw,
    },
    "erfinv-bw": {
        "tt_op": ttnn_ops.erfinv_bw,
        "pytorch_op": pytorch_ops.erfinv_bw,
    },
    "expm1-bw": {
        "tt_op": ttnn_ops.expm1_bw,
        "pytorch_op": pytorch_ops.expm1_bw,
    },
    "bitwise-and": {
        "tt_op": ttnn_ops.eltwise_bitwise_and,
        "pytorch_op": pytorch_ops.bitwise_and,
    },
    "bitwise-or": {
        "tt_op": ttnn_ops.eltwise_bitwise_or,
        "pytorch_op": pytorch_ops.bitwise_or,
    },
    "concat-bw": {
        "tt_op": ttnn_ops.concat_bw,
        "pytorch_op": ttnn_pytorch_ops.concat_bw,
    },
    "floor-bw": {
        "tt_op": ttnn_ops.floor_bw,
        "pytorch_op": pytorch_ops.floor_bw,
    },
    "eltwise-floor": {
        "tt_op": ttnn_ops.floor,
        "pytorch_op": pytorch_ops.floor,
    },
    "eltwise-unary_floor_div": {
        "tt_op": ttnn_ops.eltwise_unary_floor_div,
        "pytorch_op": pytorch_ops.unary_floor_div,
    },
    "fill-bw": {
        "tt_op": ttnn_ops.fill_bw,
        "pytorch_op": pytorch_ops.fill_bw,
    },
    "eltwise-div_no_nan": {
        "tt_op": ttnn_ops.eltwise_div_no_nan,
        "pytorch_op": pytorch_ops.div_no_nan,
    },
    "eltwise-unary_div_no_nan": {
        "tt_op": ttnn_ops.eltwise_unary_div_no_nan,
        "pytorch_op": pytorch_ops.unary_div_no_nan,
    },
    "complex-polar-bw": {
        "tt_op": ttnn_ops.complex_polar_bw,
        "pytorch_op": pytorch_ops.complex_polar_bw,
    },
    "complex-conj": {
        "tt_op": ttnn_ops.complex_conj,
        "pytorch_op": pytorch_ops.complex_conj,
    },
    "complex-abs": {
        "tt_op": ttnn_ops.complex_abs,
        "pytorch_op": pytorch_ops.complex_abs,
    },
    "complex-polar": {
        "tt_op": ttnn_ops.complex_polar,
        "pytorch_op": pytorch_ops.complex_polar,
    },
    "complex-recip": {
        "tt_op": ttnn_ops.complex_recip,
        "pytorch_op": pytorch_ops.complex_recip,
    },
    "complex-polar-bw": {
        "tt_op": ttnn_ops.complex_polar_bw,
        "pytorch_op": pytorch_ops.complex_polar_bw,
    },
    "complex-recip-bw": {
        "tt_op": ttnn_ops.complex_recip_bw,
        "pytorch_op": pytorch_ops.complex_recip_bw,
    },
    "complex-mul-bw": {
        "tt_op": ttnn_ops.complex_mul_bw,
        "pytorch_op": pytorch_ops.complex_mul_bw,
    },
    "complex-add-bw": {
        "tt_op": ttnn_ops.complex_add_bw,
        "pytorch_op": pytorch_ops.complex_add_bw,
    },
    "complex-angle": {
        "tt_op": ttnn_ops.complex_angle,
        "pytorch_op": pytorch_ops.complex_angle,
    },
    "complex-conj-bw": {
        "tt_op": ttnn_ops.complex_conj_bw,
        "pytorch_op": pytorch_ops.complex_conj_bw,
    },
    "eltwise-celu": {
        "tt_op": ttnn_ops.eltwise_celu,
        "pytorch_op": pytorch_ops.celu,
    },
    "eltwise-bias_gelu_unary": {
        "tt_op": ttnn_ops.eltwise_bias_gelu_unary,
        "pytorch_op": pytorch_ops.bias_gelu_unary,
    },
    "eltwise-identity": {
        "tt_op": ttnn_ops.eltwise_identity,
        "pytorch_op": pytorch_ops.eltwise_identity,
    },
    "unary-remainder": {
        "tt_op": ttnn_ops.unary_remainder,
        "pytorch_op": pytorch_ops.unary_remainder,
    },
    "eltwise-remainder": {
        "tt_op": ttnn_ops.eltwise_remainder,
        "pytorch_op": pytorch_ops.remainder,
    },
    "backward-unary-remainder": {
        "tt_op": ttnn_ops.unary_remainder_bw,
        "pytorch_op": pytorch_ops.unary_remainder_bw,
    },
    "topk": {
        "tt_op": ttnn_ops.topk,
        "pytorch_op": pytorch_ops.topk,
    },
    "eltwise-trunc": {
        "tt_op": ttnn_ops.eltwise_trunc,
        "pytorch_op": pytorch_ops.trunc,
    },
    "eltwise-frac": {
        "tt_op": ttnn_ops.eltwise_frac,
        "pytorch_op": pytorch_ops.frac,
    },
    "eltwise-ceil": {
        "tt_op": ttnn_ops.eltwise_ceil,
        "pytorch_op": pytorch_ops.ceil,
    },
    "argmax": {
        "tt_op": ttnn_ops.argmax,
        "pytorch_op": pytorch_ops.argmax,
    },
    "complex-imag": {
        "tt_op": ttnn_ops.complex_imag,
        "pytorch_op": pytorch_ops.complex_imag,
    },
    "complex-imag-bw": {
        "tt_op": ttnn_ops.complex_imag_bw,
        "pytorch_op": pytorch_ops.complex_imag_bw,
    },
    "complex-real": {
        "tt_op": ttnn_ops.complex_real,
        "pytorch_op": pytorch_ops.complex_real,
    },
    "complex-real-bw": {
        "tt_op": ttnn_ops.complex_real_bw,
        "pytorch_op": pytorch_ops.complex_real_bw,
    },
    "complex-angle-bw": {
        "tt_op": ttnn_ops.complex_angle_bw,
        "pytorch_op": pytorch_ops.complex_angle_bw,
    },
    "complex-is-imag": {
        "tt_op": ttnn_ops.complex_is_imag,
        "pytorch_op": pytorch_ops.complex_is_imag,
    },
    "complex-is-real": {
        "tt_op": ttnn_ops.complex_is_real,
        "pytorch_op": pytorch_ops.complex_is_real,
    },
    "relu6-bw": {
        "tt_op": ttnn_ops.relu6_bw,
        "pytorch_op": pytorch_ops.relu6_bw,
    },
}
