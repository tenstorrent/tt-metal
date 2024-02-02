APIs
####

Device
******

.. toctree::
   :maxdepth: 1

   ttnn/open_device
   ttnn/close_device
   ttnn/manage_device


Tensor
******

.. toctree::
   :maxdepth: 1

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


Memory Config
*************

.. toctree::
   :maxdepth: 1

   ttnn/create_sharded_memory_config


Operations
**********

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

   ttnn/exp
   ttnn/gelu
   ttnn/log
   ttnn/relu
   ttnn/rsqrt
   ttnn/silu
   ttnn/softmax
   ttnn/tanh
   ttnn/sin
   ttnn/cos
   ttnn/tan
   ttnn/asin
   ttnn/acos
   ttnn/atan
   ttnn/sinh
   ttnn/cosh
   ttnn/asinh
   ttnn/acosh
   ttnn/atanh
   ttnn/logical_not
   ttnn/logit
   ttnn/clone

Pointwise Binary
================

.. toctree::
   :maxdepth: 1

   ttnn/add
   ttnn/mul
   ttnn/sub
   ttnn/pow
   ttnn/add_and_apply_activation
   ttnn/add_and_apply_activation_

Pointwise Relational
====================

.. toctree::
   :maxdepth: 1

   ttnn/gtz
   ttnn/ltz
   ttnn/gez
   ttnn/lez
   ttnn/nez
   ttnn/eqz
   ttnn/gt
   ttnn/gte
   ttnn/lt
   ttnn/lte
   ttnn/eq
   ttnn/ne
   ttnn/isclose

Pointwise Math
==============

.. toctree::
   :maxdepth: 1

   ttnn/i0
   ttnn/isfinite
   ttnn/isinf
   ttnn/isnan
   ttnn/isneginf
   ttnn/isposinf
   ttnn/lgamma
   ttnn/log10
   ttnn/log1p
   ttnn/log2
   ttnn/multigammaln
   ttnn/neg
   ttnn/abs
   ttnn/cbrt
   ttnn/deg2rad
   ttnn/digamma
   ttnn/erf
   ttnn/erfc
   ttnn/erfinv
   ttnn/exp2
   ttnn/expm1
   ttnn/atan2
   ttnn/hypot
   ttnn/lerp
   ttnn/squared_difference

Activation
==========

.. toctree::
   :maxdepth: 1

   ttnn/clip
   ttnn/elu
   ttnn/hardshrink
   ttnn/hardswish
   ttnn/hardtanh
   ttnn/heaviside
   ttnn/leaky_relu
   ttnn/log_sigmoid
   ttnn/mish
   ttnn/prelu
   ttnn/relu_max
   ttnn/relu_min
   ttnn/relu6
   ttnn/sigmoid
   ttnn/sign
   ttnn/softshrink
   ttnn/softsign
   ttnn/swish
   ttnn/hardsigmoid
   ttnn/softplus

Tensor Creation
===============

.. toctree::
   :maxdepth: 1

   ttnn/ones
   ttnn/ones_like
   ttnn/zeros
   ttnn/zeros_like
   ttnn/full
   ttnn/full_like

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

   ttnn/mean
   ttnn/std
   ttnn/var

Data Movement
=============

.. toctree::
   :maxdepth: 1

   ttnn/concat
   ttnn/pad
   ttnn/permute
   ttnn/reshape
   ttnn/split
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

Sampling
========

.. toctree::
   :maxdepth: 1

   ttnn/upsample


Model Conversion
****************

.. toctree::
   :maxdepth: 1

   ttnn/model_preprocessing/preprocess_model
   ttnn/model_preprocessing/preprocess_model_parameters
