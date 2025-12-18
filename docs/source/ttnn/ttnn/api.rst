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
   ttnn.copy_device_to_host_tensor
   ttnn.copy_host_to_device_tensor
   ttnn.deallocate
   ttnn.dump_tensor
   ttnn.from_device
   ttnn.from_torch
   ttnn.get_device_tensors
   ttnn.load_tensor
   ttnn.reallocate
   ttnn.split_work_to_cores
   ttnn.to_device
   ttnn.to_dtype
   ttnn.to_layout
   ttnn.to_memory_config
   ttnn.to_torch
   ttnn.typecast

Tensor Creation
===============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.arange
   ttnn.bernoulli
   ttnn.complex_tensor
   ttnn.empty
   ttnn.empty_like
   ttnn.from_buffer
   ttnn.full
   ttnn.full_like
   ttnn.index_fill
   ttnn.ones
   ttnn.ones_like
   ttnn.rand
   ttnn.uniform
   ttnn.zeros
   ttnn.zeros_like

Matrix Multiplication
=====================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.matmul
   ttnn.linear
   ttnn.addmm
   ttnn.sparse_matmul

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: class.rst

   ttnn.MatmulMultiCoreReuseProgramConfig
   ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
   ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
   ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig


Pointwise Unary
================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.abs
   ttnn.abs_bw
   ttnn.acos
   ttnn.acos_bw
   ttnn.acosh
   ttnn.acosh_bw
   ttnn.alt_complex_rotate90
   ttnn.angle
   ttnn.angle_bw
   ttnn.asin
   ttnn.asin_bw
   ttnn.asinh
   ttnn.asinh_bw
   ttnn.atan
   ttnn.atan_bw
   ttnn.atanh
   ttnn.atanh_bw
   ttnn.bitcast
   ttnn.bitwise_left_shift
   ttnn.bitwise_not
   ttnn.bitwise_right_shift
   ttnn.cbrt
   ttnn.ceil
   ttnn.ceil_bw
   ttnn.celu
   ttnn.celu_bw
   ttnn.clamp
   ttnn.clamp_bw
   ttnn.clip
   ttnn.clip_bw
   ttnn.clone
   ttnn.conj
   ttnn.conj_bw
   ttnn.cos
   ttnn.cos_bw
   ttnn.cosh
   ttnn.cosh_bw
   ttnn.deg2rad
   ttnn.deg2rad_bw
   ttnn.digamma
   ttnn.digamma_bw
   ttnn.div_no_nan_bw
   ttnn.elu_bw
   ttnn.eqz
   ttnn.erf
   ttnn.erf_bw
   ttnn.erfc
   ttnn.erfc_bw
   ttnn.erfinv
   ttnn.erfinv_bw
   ttnn.exp
   ttnn.exp2
   ttnn.exp2_bw
   ttnn.exp_bw
   ttnn.experimental.dropout
   ttnn.experimental.gelu_bw
   ttnn.elu
   ttnn.expm1
   ttnn.expm1_bw
   ttnn.fill
   ttnn.fill_bw
   ttnn.fill_zero_bw
   ttnn.floor
   ttnn.floor_bw
   ttnn.frac
   ttnn.frac_bw
   ttnn.geglu
   ttnn.gelu
   ttnn.gelu_bw
   ttnn.gez
   ttnn.glu
   ttnn.gtz
   ttnn.hardmish
   ttnn.hardshrink
   ttnn.hardshrink_bw
   ttnn.hardsigmoid
   ttnn.hardsigmoid_bw
   ttnn.hardswish
   ttnn.hardswish_bw
   ttnn.hardtanh
   ttnn.hardtanh_bw
   ttnn.heaviside
   ttnn.i0
   ttnn.i0_bw
   ttnn.i1
   ttnn.identity
   ttnn.imag
   ttnn.imag_bw
   ttnn.is_imag
   ttnn.is_real
   ttnn.isfinite
   ttnn.isinf
   ttnn.isnan
   ttnn.isneginf
   ttnn.isposinf
   ttnn.leaky_relu
   ttnn.leaky_relu_bw
   ttnn.lez
   ttnn.lgamma
   ttnn.lgamma_bw
   ttnn.log
   ttnn.log10
   ttnn.log10_bw
   ttnn.log1p
   ttnn.log1p_bw
   ttnn.log2
   ttnn.log2_bw
   ttnn.log_bw
   ttnn.log_sigmoid
   ttnn.log_sigmoid_bw
   ttnn.logical_left_shift
   ttnn.logical_not
   ttnn.logical_not_
   ttnn.logical_right_shift
   ttnn.logit
   ttnn.logit_bw
   ttnn.logiteps_bw
   ttnn.ltz
   ttnn.mish
   ttnn.multigammaln
   ttnn.multigammaln_bw
   ttnn.neg
   ttnn.neg_bw
   ttnn.nez
   ttnn.normalize_global
   ttnn.normalize_hw
   ttnn.polar
   ttnn.polar_bw
   ttnn.polygamma
   ttnn.polygamma_bw
   ttnn.pow_bw
   ttnn.prelu
   ttnn.prod_bw
   ttnn.rad2deg
   ttnn.rad2deg_bw
   ttnn.rdiv
   ttnn.rdiv_bw
   ttnn.real
   ttnn.real_bw
   ttnn.reciprocal
   ttnn.reciprocal_bw
   ttnn.reglu
   ttnn.relu
   ttnn.relu6
   ttnn.relu6_bw
   ttnn.relu_bw
   ttnn.relu_max
   ttnn.relu_min
   ttnn.remainder
   ttnn.repeat_bw
   ttnn.round
   ttnn.round_bw
   ttnn.rpow_bw
   ttnn.rsqrt
   ttnn.rsqrt_bw
   ttnn.selu
   ttnn.selu_bw
   ttnn.sigmoid
   ttnn.sigmoid_accurate
   ttnn.sigmoid_bw
   ttnn.sign
   ttnn.sign_bw
   ttnn.signbit
   ttnn.silu
   ttnn.silu_bw
   ttnn.sin
   ttnn.sin_bw
   ttnn.sinh
   ttnn.sinh_bw
   ttnn.softplus
   ttnn.softplus_bw
   ttnn.softshrink
   ttnn.softshrink_bw
   ttnn.softsign
   ttnn.softsign_bw
   ttnn.sqrt
   ttnn.sqrt_bw
   ttnn.square
   ttnn.square_bw
   ttnn.std_hw
   ttnn.swiglu
   ttnn.swish
   ttnn.tan
   ttnn.tan_bw
   ttnn.tanh
   ttnn.tanh_bw
   ttnn.tanhshrink
   ttnn.tanhshrink_bw
   ttnn.threshold
   ttnn.threshold_bw
   ttnn.tril
   ttnn.triu
   ttnn.trunc
   ttnn.trunc_bw
   ttnn.unary_chain
   ttnn.var_hw

Pointwise Binary
================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.add
   ttnn.add_
   ttnn.add_bw
   ttnn.addalpha
   ttnn.addalpha_bw
   ttnn.assign_bw
   ttnn.atan2
   ttnn.atan2_bw
   ttnn.bias_gelu
   ttnn.bias_gelu_
   ttnn.bias_gelu_bw
   ttnn.bitwise_and
   ttnn.bitwise_or
   ttnn.bitwise_xor
   ttnn.concat_bw
   ttnn.div
   ttnn.div_bw
   ttnn.div_no_nan
   ttnn.divide
   ttnn.divide_
   ttnn.embedding_bw
   ttnn.eq
   ttnn.eq_
   ttnn.floor_div
   ttnn.fmod
   ttnn.fmod_bw
   ttnn.gcd
   ttnn.ge
   ttnn.ge_
   ttnn.gt
   ttnn.gt_
   ttnn.hypot
   ttnn.hypot_bw
   ttnn.isclose
   ttnn.lcm
   ttnn.ldexp
   ttnn.ldexp_
   ttnn.ldexp_bw
   ttnn.le
   ttnn.le_
   ttnn.logaddexp
   ttnn.logaddexp2
   ttnn.logaddexp2_
   ttnn.logaddexp2_bw
   ttnn.logaddexp_
   ttnn.logaddexp_bw
   ttnn.logical_and
   ttnn.logical_and_
   ttnn.logical_or
   ttnn.logical_or_
   ttnn.logical_xor
   ttnn.logical_xor_
   ttnn.lt
   ttnn.lt_
   ttnn.max_bw
   ttnn.maximum
   ttnn.min_bw
   ttnn.minimum
   ttnn.mul_bw
   ttnn.multiply
   ttnn.multiply_
   ttnn.ne
   ttnn.ne_
   ttnn.nextafter
   ttnn.outer
   ttnn.polyval
   ttnn.pow
   ttnn.prim.binary
   ttnn.remainder
   ttnn.remainder_bw
   ttnn.rpow
   ttnn.rsub
   ttnn.rsub_
   ttnn.rsub_bw
   ttnn.scatter
   ttnn.scatter_add
   ttnn.squared_difference
   ttnn.squared_difference_
   ttnn.squared_difference_bw
   ttnn.sub_bw
   ttnn.subalpha
   ttnn.subalpha_bw
   ttnn.subtract
   ttnn.subtract_
   ttnn.xlogy
   ttnn.xlogy_bw

Pointwise Ternary
=================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.addcdiv
   ttnn.addcdiv_bw
   ttnn.addcmul
   ttnn.addcmul_bw
   ttnn.lerp
   ttnn.lerp_bw
   ttnn.mac
   ttnn.where
   ttnn.where_bw

Quantization
============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.dequantize
   ttnn.quantize
   ttnn.requantize

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

   ttnn.argmax
   ttnn.cumprod
   ttnn.cumsum
   ttnn.ema
   ttnn.manual_seed
   ttnn.max
   ttnn.mean
   ttnn.min
   ttnn.moe
   ttnn.prod
   ttnn.sampling
   ttnn.std
   ttnn.sum
   ttnn.topk
   ttnn.var

Data Movement
=============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.assign
   ttnn.bcast
   ttnn.chunk
   ttnn.concat
   ttnn.copy
   ttnn.expand
   ttnn.fill_implicit_tile_padding
   ttnn.fill_ones_rm
   ttnn.fill_rm
   ttnn.fold
   ttnn.gather
   ttnn.indexed_fill
   ttnn.interleaved_to_sharded
   ttnn.interleaved_to_sharded_partial
   ttnn.moe_expert_token_remap
   ttnn.moe_routing_remap
   ttnn.move
   ttnn.nonzero
   ttnn.pad
   ttnn.permute
   ttnn.repeat
   ttnn.repeat_interleave
   ttnn.reshape
   ttnn.reshape_on_device
   ttnn.reshard
   ttnn.roll
   ttnn.sharded_to_interleaved
   ttnn.sharded_to_interleaved_partial
   ttnn.slice
   ttnn.sort
   ttnn.split
   ttnn.squeeze
   ttnn.stack
   ttnn.tilize
   ttnn.tilize_with_val_padding
   ttnn.tilize_with_zero_padding
   ttnn.transpose
   ttnn.unsqueeze
   ttnn.unsqueeze_to_4D
   ttnn.untilize
   ttnn.untilize_with_unpadding
   ttnn.view

Normalization
=============

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.batch_norm
   ttnn.group_norm
   ttnn.layer_norm
   ttnn.layer_norm_post_all_gather
   ttnn.layer_norm_pre_all_gather
   ttnn.rms_norm
   ttnn.rms_norm_post_all_gather
   ttnn.rms_norm_pre_all_gather
   ttnn.scale_causal_mask_hw_dims_softmax_in_place
   ttnn.scale_mask_softmax
   ttnn.scale_mask_softmax_in_place
   ttnn.softmax
   ttnn.softmax_in_place

Normalization Program Configs
=============================

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: class.rst

   ttnn.SoftmaxDefaultProgramConfig
   ttnn.SoftmaxProgramConfig
   ttnn.SoftmaxShardedMultiCoreProgramConfig

Transformer
===========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.transformer.attention_softmax
   ttnn.transformer.attention_softmax_
   ttnn.transformer.chunked_flash_mla_prefill
   ttnn.transformer.chunked_scaled_dot_product_attention
   ttnn.transformer.concatenate_heads
   ttnn.transformer.flash_mla_prefill
   ttnn.transformer.flash_multi_latent_attention_decode
   ttnn.transformer.joint_scaled_dot_product_attention
   ttnn.transformer.paged_flash_multi_latent_attention_decode
   ttnn.transformer.paged_scaled_dot_product_attention_decode
   ttnn.transformer.ring_distributed_scaled_dot_product_attention
   ttnn.transformer.ring_joint_scaled_dot_product_attention
   ttnn.transformer.scaled_dot_product_attention
   ttnn.transformer.scaled_dot_product_attention_decode
   ttnn.transformer.split_query_key_value_and_split_heads
   ttnn.transformer.windowed_scaled_dot_product_attention

CCL
===

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.all_broadcast
   ttnn.all_gather
   ttnn.all_reduce
   ttnn.all_to_all_combine
   ttnn.all_to_all_dispatch
   ttnn.broadcast
   ttnn.mesh_partition
   ttnn.point_to_point
   ttnn.reduce_scatter
   ttnn.reduce_to_root

Embedding
=========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.embedding

Convolution
===========
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.conv1d
   ttnn.conv2d
   ttnn.conv_transpose2d
   ttnn.experimental.conv3d
   ttnn.prepare_conv_bias
   ttnn.prepare_conv_transpose2d_bias
   ttnn.prepare_conv_transpose2d_weights
   ttnn.prepare_conv_weights

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: class.rst

   ttnn.Conv2dConfig
   ttnn.Conv2dSliceConfig

Pooling
=======

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.adaptive_avg_pool2d
   ttnn.adaptive_max_pool2d
   ttnn.avg_pool2d
   ttnn.global_avg_pool2d
   ttnn.grid_sample
   ttnn.max_pool2d

Prefetcher
==========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.dram_prefetcher

Vision
========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.upsample

Generic
=======

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.generic_op

KV Cache
========

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: function.rst

   ttnn.kv_cache.fill_cache_for_user_
   ttnn.kv_cache.update_cache_for_token_
   ttnn.fill_cache
   ttnn.update_cache


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
