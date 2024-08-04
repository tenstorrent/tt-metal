APIs
####

Device
******

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.open_device
   ttnn.close_device
   ttnn.manage_device
   ttnn.synchronize_device
   ttnn.SetDefaultDevice
   ttnn.GetDefaultDevice
   ttnn.format_input_tensor
   ttnn.format_output_tensor
   ttnn.pad_to_tile_shape

Memory Config
*************

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.create_sharded_memory_config


Operations
**********

Core
====

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.as_tensor
   ttnn.from_torch
   ttnn.to_torch
   ttnn.to_device
   ttnn.from_device
   ttnn.to_layout
   ttnn.dump_tensor
   ttnn.load_tensor
   ttnn.deallocate
   ttnn.reallocate
   ttnn.to_memory_config


Tensor Creation
===============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.arange
   ttnn.empty
   ttnn.empty_like
   ttnn.zeros
   ttnn.zeros_like
   ttnn.ones
   ttnn.ones_like
   ttnn.full
   ttnn.full_like

Matrix Multiplication
=====================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.matmul
   ttnn.linear

Pointwise Unary
================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.abs
   ttnn.acos
   ttnn.logical_not_
   ttnn.acosh
   ttnn.asin
   ttnn.asinh
   ttnn.atan
   ttnn.atanh
   ttnn.bitwise_and
   ttnn.bitwise_or
   ttnn.bitwise_xor
   ttnn.bitwise_not
   ttnn.bitwise_left_shift
   ttnn.bitwise_right_shift
   ttnn.cbrt
   ttnn.celu
   ttnn.clamp
   ttnn.clip
   ttnn.clone
   ttnn.cos
   ttnn.cosh
   ttnn.deg2rad
   ttnn.digamma
   ttnn.elu
   ttnn.erf
   ttnn.erfc
   ttnn.erfinv
   ttnn.exp
   ttnn.exp2
   ttnn.expm1
   ttnn.floor
   ttnn.ceil
   ttnn.geglu
   ttnn.gelu
   ttnn.glu
   ttnn.hardshrink
   ttnn.normalize_global
   ttnn.hardsigmoid
   ttnn.hardswish
   ttnn.hardtanh
   ttnn.heaviside
   ttnn.hypot
   ttnn.i0
   ttnn.identity
   ttnn.isfinite
   ttnn.isinf
   ttnn.isnan
   ttnn.isneginf
   ttnn.isposinf
   ttnn.leaky_relu
   ttnn.lerp
   ttnn.lgamma
   ttnn.log
   ttnn.log10
   ttnn.log1p
   ttnn.log2
   ttnn.log_sigmoid
   ttnn.logical_not
   ttnn.frac
   ttnn.logit
   ttnn.mish
   ttnn.multigammaln
   ttnn.neg
   ttnn.normalize_hw
   ttnn.prelu
   ttnn.reglu
   ttnn.relu
   ttnn.relu_max
   ttnn.relu_min
   ttnn.relu6
   ttnn.remainder
   ttnn.rsqrt
   ttnn.rdiv
   ttnn.rsub
   ttnn.selu
   ttnn.sigmoid
   ttnn.sigmoid_accurate
   ttnn.sign
   ttnn.silu
   ttnn.sin
   ttnn.sinh
   ttnn.softmax
   ttnn.softplus
   ttnn.softshrink
   ttnn.softsign
   ttnn.swish
   ttnn.tan
   ttnn.tanh
   ttnn.signbit
   ttnn.polygamma
   ttnn.rad2deg
   ttnn.reciprocal
   ttnn.round
   ttnn.sqrt
   ttnn.square
   ttnn.swiglu
   ttnn.tril
   ttnn.triu
   ttnn.tanhshrink
   ttnn.threshold
   ttnn.trunc
   ttnn.mul_bw
   ttnn.clamp_bw
   ttnn.hardtanh_bw
   ttnn.threshold_bw
   ttnn.softplus_bw
   ttnn.div_bw
   ttnn.rdiv_bw
   ttnn.bias_gelu_bw
   ttnn.pow_bw
   ttnn.exp_bw
   ttnn.tanh_bw
   ttnn.sqrt_bw
   ttnn.assign_bw
   ttnn.multigammaln_bw
   ttnn.add_bw
   ttnn.lgamma_bw
   ttnn.fill_bw
   ttnn.hardsigmoid_bw
   ttnn.cos_bw
   ttnn.acosh_bw
   ttnn.acos_bw
   ttnn.atan_bw
   ttnn.rad2deg_bw
   ttnn.sub_bw
   ttnn.frac_bw
   ttnn.trunc_bw
   ttnn.log_sigmoid_bw
   ttnn.fill_zero_bw
   ttnn.i0_bw
   ttnn.tan_bw
   ttnn.sigmoid_bw
   ttnn.rsqrt_bw
   ttnn.neg_bw
   ttnn.relu_bw
   ttnn.logit_bw
   ttnn.hardshrink_bw
   ttnn.softshrink_bw
   ttnn.leaky_relu_bw
   ttnn.elu_bw
   ttnn.celu_bw
   ttnn.rpow_bw
   ttnn.floor_bw
   ttnn.round_bw
   ttnn.log_bw
   ttnn.relu6_bw
   ttnn.abs_bw
   ttnn.silu_bw
   ttnn.selu_bw
   ttnn.square_bw
   ttnn.prod_bw
   ttnn.hardswish_bw
   ttnn.tanhshrink_bw
   ttnn.atanh_bw
   ttnn.asin_bw
   ttnn.asinh_bw
   ttnn.sin_bw
   ttnn.sinh_bw
   ttnn.log10_bw
   ttnn.log1p_bw
   ttnn.erfc_bw
   ttnn.ceil_bw
   ttnn.softsign_bw
   ttnn.cosh_bw
   ttnn.logiteps_bw
   ttnn.log2_bw
   ttnn.sign_bw
   ttnn.fmod_bw
   ttnn.remainder_bw
   ttnn.div_no_nan_bw
   ttnn.exp2_bw
   ttnn.expm1_bw
   ttnn.reciprocal_bw
   ttnn.digamma_bw
   ttnn.erfinv_bw
   ttnn.erf_bw
   ttnn.deg2rad_bw
   ttnn.polygamma_bw
   ttnn.gelu_bw
   ttnn.repeat_bw
   ttnn.real
   ttnn.imag
   ttnn.angle
   ttnn.is_imag
   ttnn.is_real
   ttnn.polar_bw
   ttnn.imag_bw
   ttnn.real_bw
   ttnn.angle_bw
   ttnn.conj_bw
   ttnn.conj
   ttnn.polar

Pointwise Binary
================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.add
   ttnn.addalpha
   ttnn.subalpha
   ttnn.multiply
   ttnn.subtract
   ttnn.div
   ttnn.div_no_nan
   ttnn.floor_div
   ttnn.remainder
   ttnn.fmod
   ttnn/gcd
   ttnn/lcm
   ttnn.logical_and_
   ttnn.logical_or_
   ttnn.logical_xor_
   ttnn.pow
   ttnn.rpow
   ttnn.ldexp
   ttnn.logical_and
   ttnn.logical_or
   ttnn.logical_xor
   ttnn.logaddexp
   ttnn.logaddexp2
   ttnn.xlogy
   ttnn.squared_difference
   ttnn.gtz
   ttnn.ltz
   ttnn.gez
   ttnn.lez
   ttnn.nez
   ttnn.eqz
   ttnn.gt
   ttnn.gt_
   ttnn.lt_
   ttnn.ge_
   ttnn.le_
   ttnn.eq_
   ttnn.ne_
   ttnn.ge
   ttnn.lt
   ttnn.le
   ttnn.eq
   ttnn.ne
   ttnn.isclose
   ttnn.nextafter
   ttnn.maximum
   ttnn.minimum
   ttnn.outer
   ttnn.polyval
   ttnn.scatter
   ttnn.atan2
   ttnn.atan2_bw
   ttnn.embedding_bw
   ttnn.addalpha_bw
   ttnn.subalpha_bw
   ttnn.xlogy_bw
   ttnn.hypot_bw
   ttnn.ldexp_bw
   ttnn.logaddexp_bw
   ttnn.logaddexp2_bw
   ttnn.squared_difference_bw
   ttnn.concat_bw
   ttnn.rsub_bw
   ttnn.min_bw
   ttnn.max_bw
   ttnn.lerp_bw

Pointwise Ternary
=================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.addcdiv
   ttnn.addcmul
   ttnn.mac
   ttnn.where
   ttnn.addcmul_bw
   ttnn.addcdiv_bw
   ttnn.where_bw

Losses
======

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.l1_loss
   ttnn.mse_loss

Reduction
=========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.max
   ttnn.mean
   ttnn.min
   ttnn.std
   ttnn.sum
   ttnn.var
   ttnn.argmax
   ttnn.prod
   ttnn.topk

Data Movement
=============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.concat
   ttnn.nonzero
   ttnn.pad
   ttnn.permute
   ttnn.reshape
   ttnn.repeat
   ttnn.repeat_interleave
   ttnn.tilize
   ttnn.tilize_with_val_padding
   ttnn.fill_rm
   ttnn.fill_ones_rm
   ttnn.untilize
   ttnn.untilize_with_unpadding
   ttnn.untilize_with_halo_v2
   ttnn.indexed_fill

Normalization
=============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.group_norm
   ttnn.layer_norm
   ttnn.rms_norm

Transformer
===========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.transformer.split_query_key_value_and_split_heads
   ttnn.transformer.concatenate_heads
   ttnn.transformer.attention_softmax
   ttnn.transformer.attention_softmax_
   ttnn.experimental.rotary_embedding
   ttnn.transformer.scaled_dot_product_attention
   ttnn.transformer.scaled_dot_product_attention_decode
   ttnn.transformer.scaled_dot_product_attention_decode_gqa

Embedding
=========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.embedding

Pooling
=======

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.global_avg_pool2d
   ttnn.max_pool2d

Vision
========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.upsample
   ttnn.downsample

KV Cache
========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.kv_cache.fill_cache_for_user_
   ttnn.kv_cache.update_cache_for_token_


Model Conversion
****************

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.model_preprocessing.preprocess_model
   ttnn.model_preprocessing.preprocess_model_parameters


Reports
*******

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.set_printoptions


Operation Hooks
***************

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.register_pre_operation_hook
   ttnn.register_post_operation_hook
