APIs
####

Device
******

.. toctree::
   :maxdepth: 1

   ttnn/open_device
   ttnn/close_device
   ttnn/manage_device
   ttnn/synchronize_device

Memory Config
*************

.. toctree::
   :maxdepth: 1

   ttnn/create_sharded_memory_config


Operations
**********

Core
====

.. toctree::
   :maxdepth: 1

   ttnn/as_tensor
   ttnn/from_torch
   ttnn/to_torch
   ttnn/to_device
   ttnn/from_device
   ttnn/to_layout
   ttnn/dump_tensor
   ttnn/load_tensor
   ttnn/deallocate
   ttnn/reallocate
   ttnn/to_memory_config


Tensor Creation
===============

.. toctree::
   :maxdepth: 1

   ttnn/arange
   ttnn/empty
   ttnn/zeros
   ttnn/zeros_like
   ttnn/ones
   ttnn/ones_like
   ttnn/full
   ttnn/full_like

Matrix Multiplication
=====================

.. toctree::
   :maxdepth: 1

   ttnn/matmul
   ttnn/linear

Pointwise Unary
================

.. toctree::
   :maxdepth: 1

   ttnn/abs
   ttnn/acos
   ttnn/acosh
   ttnn/asin
   ttnn/asinh
   ttnn/atan
   ttnn/atan2
   ttnn/atanh
   ttnn/bitwise_and
   ttnn/bitwise_or
   ttnn/bitwise_xor
   ttnn/bitwise_not
   ttnn/bitwise_left_shift
   ttnn/bitwise_right_shift
   ttnn/cbrt
   ttnn/celu
   ttnn/clip
   ttnn/clone
   ttnn/cos
   ttnn/cosh
   ttnn/deg2rad
   ttnn/digamma
   ttnn/elu
   ttnn/erf
   ttnn/erfc
   ttnn/erfinv
   ttnn/exp
   ttnn/exp2
   ttnn/expm1
   ttnn/floor
   ttnn/geglu
   ttnn/gelu
   ttnn/glu
   ttnn/hardshrink
   ttnn/hardsigmoid
   ttnn/hardswish
   ttnn/hardtanh
   ttnn/heaviside
   ttnn/hypot
   ttnn/i0
   ttnn/identity
   ttnn/isfinite
   ttnn/isinf
   ttnn/isnan
   ttnn/isneginf
   ttnn/isposinf
   ttnn/leaky_relu
   ttnn/lerp
   ttnn/lgamma
   ttnn/log
   ttnn/log10
   ttnn/log1p
   ttnn/log2
   ttnn/log_sigmoid
   ttnn/logical_not
   ttnn/logit
   ttnn/mish
   ttnn/multigammaln
   ttnn/neg
   ttnn/prelu
   ttnn/reglu
   ttnn/relu
   ttnn/relu_max
   ttnn/relu_min
   ttnn/relu6
   ttnn/remainder
   ttnn/rsqrt
   ttnn/rdiv
   ttnn/rsub
   ttnn/sigmoid
   ttnn/sigmoid_accurate
   ttnn/sign
   ttnn/silu
   ttnn/sin
   ttnn/sinh
   ttnn/softmax
   ttnn/softplus
   ttnn/softshrink
   ttnn/softsign
   ttnn/swish
   ttnn/tan
   ttnn/tanh
   ttnn/signbit
   ttnn/polygamma
   ttnn/rad2deg
   ttnn/reciprocal
   ttnn/sqrt
   ttnn/square
   ttnn/swiglu
   ttnn/tril
   ttnn/triu
   ttnn/tanhshrink
   ttnn/threshold
   ttnn/mul_bw
   ttnn/clamp_min_bw
   ttnn/clamp_bw
   ttnn/assign_bw
   ttnn/multigammaln_bw
   ttnn/add_bw
   ttnn/eq_bw
   ttnn/lgamma_bw
   ttnn/unary_sub_bw

Pointwise Binary
================

.. toctree::
   :maxdepth: 1

   ttnn/add
   ttnn/multiply
   ttnn/subtract
   ttnn/pow
   ttnn/ldexp
   ttnn/logical_and
   ttnn/logical_or
   ttnn/logical_xor
   ttnn/logaddexp
   ttnn/logaddexp2
   ttnn/xlogy
   ttnn/squared_difference
   ttnn/gtz
   ttnn/ltz
   ttnn/gez
   ttnn/lez
   ttnn/nez
   ttnn/eqz
   ttnn/gt
   ttnn/ge
   ttnn/lt
   ttnn/le
   ttnn/eq
   ttnn/ne
   ttnn/isclose
   ttnn/polyval
   ttnn/nextafter
   ttnn/maximum
   ttnn/minimum
   ttnn/atan2_bw
   ttnn/embedding_bw
   ttnn/addalpha_bw
   ttnn/subalpha_bw
   ttnn/sub_bw
   ttnn/xlogy_bw
   ttnn/hypot_bw
   ttnn/ldexp_bw
   ttnn/logaddexp_bw
   ttnn/logaddexp2_bw
   ttnn/squared_difference_bw
   ttnn/concat_bw
   ttnn/binary_le_bw
   ttnn/rsub_bw
   ttnn/bias_gelu_bw
   ttnn/binary_gt_bw
   ttnn/binary_lt_bw
   ttnn/binary_ne_bw
   ttnn/binary_ge_bw
   ttnn/min_bw
   ttnn/max_bw
   ttnn/div_bw
   ttnn/lerp_bw

Pointwise Ternary
=================

.. toctree::
   :maxdepth: 1

   ttnn/addcdiv
   ttnn/addcmul
   ttnn/mac
   ttnn/where
   ttnn/addcmul_bw
   ttnn/addcdiv_bw
   ttnn/where_bw

Losses
======

.. toctree::
   :maxdepth: 1

   ttnn/l1_loss
   ttnn/mse_loss

Reduction
=========

.. toctree::
   :maxdepth: 1

   ttnn/max
   ttnn/mean
   ttnn/min
   ttnn/std
   ttnn/sum
   ttnn/var
   ttnn/argmax
   ttnn/topk

Data Movement
=============

.. toctree::
   :maxdepth: 1

   ttnn/concat
   ttnn/pad
   ttnn/permute
   ttnn/reshape
   ttnn/repeat
   ttnn/repeat_interleave

Normalization
=============

.. toctree::
   :maxdepth: 1

   ttnn/group_norm
   ttnn/layer_norm
   ttnn/rms_norm

Transformer
===========

.. toctree::
   :maxdepth: 1

   ttnn/transformer/split_query_key_value_and_split_heads
   ttnn/transformer/concatenate_heads
   ttnn/transformer/attention_softmax
   ttnn/transformer/attention_softmax_
   ttnn/transformer/rotary_embedding

Embedding
=========

.. toctree::
   :maxdepth: 1

   ttnn/embedding

Pooling
=======

.. toctree::
   :maxdepth: 1

   ttnn/global_avg_pool2d
   ttnn/MaxPool2d

Vision
========

.. toctree::
   :maxdepth: 1

   ttnn/upsample

KV Cache
========

.. toctree::
   :maxdepth: 1

   ttnn/kv_cache/fill_cache_for_user_
   ttnn/kv_cache/update_cache_for_token_


Model Conversion
****************

.. toctree::
   :maxdepth: 1

   ttnn/model_preprocessing/preprocess_model
   ttnn/model_preprocessing/preprocess_model_parameters


Reports
*******
.. toctree::
   :maxdepth: 1

   ttnn/set_printoptions


Operation Hooks
***************
.. toctree::
   :maxdepth: 1

   ttnn/register_pre_operation_hook
   ttnn/register_post_operation_hook
