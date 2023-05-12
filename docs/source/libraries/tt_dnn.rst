.. _TT-DNN:

TT-DNN
******

Overview
========

TT-DNN is a simplified Python interface to the compute engine of the TT-Metal.

This will be the future plan. For now, the ``tt_lib`` Python module is a
unified Python interface that provides both TT-DNN and the Tensor library.

tt-DNN API through ``tt_lib``
=============================

Enums
-----

.. autoclass:: tt_lib.tensor.BcastOpMath
    :members: ADD, SUB, MUL

.. autoclass:: tt_lib.tensor.BcastOpDim
    :members: H, W, HW

.. autoclass:: tt_lib.tensor.ReduceOpMath
    :members: SUM, MAX

.. autoclass:: tt_lib.tensor.ReduceOpDim
    :members: H, W, HW

Tensor elementwise operations
-----------------------------

.. autofunction:: tt_lib.tensor.add

.. autofunction:: tt_lib.tensor.sub

.. autofunction:: tt_lib.tensor.mul

.. autofunction:: tt_lib.tensor.gelu

.. autofunction:: tt_lib.tensor.relu

..
    autofunction:: tt_lib.tensor.sigmoid

.. autofunction:: tt_lib.tensor.exp

.. autofunction:: tt_lib.tensor.recip

.. autofunction:: tt_lib.tensor.sqrt

.. autofunction:: tt_lib.tensor.log

.. autofunction:: tt_lib.tensor.tanh


Tensor matrix math operations
-----------------------------

.. autofunction:: tt_lib.tensor.matmul

.. autofunction:: tt_lib.tensor.bmm

Tensor manipulation operations
------------------------------

These operations change the tensor shape in some way, giving it new dimensions
but in general retaining the data.

.. autofunction:: tt_lib.tensor.reshape

.. autofunction:: tt_lib.tensor.transpose

.. autofunction:: tt_lib.tensor.transpose_hc

.. autofunction:: tt_lib.tensor.permute


Broadcast and Reduce
--------------------

.. autofunction:: tt_lib.tensor.bcast

.. autofunction:: tt_lib.tensor.reduce

Fallback Operations
===================

These operations are currently not supported on TT accelerator device and will execute on host machine using Pytorch.

.. autofunction:: tt_lib.fallback_ops.fallback_ops.full

.. autofunction:: tt_lib.fallback_ops.fallback_ops.reshape

.. autofunction:: tt_lib.fallback_ops.fallback_ops.chunk

.. autofunction:: tt_lib.fallback_ops.fallback_ops.conv2d

.. autofunction:: tt_lib.fallback_ops.fallback_ops.group_norm

.. autofunction:: tt_lib.fallback_ops.fallback_ops.layer_norm

.. autofunction:: tt_lib.fallback_ops.fallback_ops.pad

.. autofunction:: tt_lib.fallback_ops.fallback_ops.repeat_interleave

.. autofunction:: tt_lib.fallback_ops.fallback_ops.concat

.. autofunction:: tt_lib.fallback_ops.fallback_ops.sigmoid

.. autofunction:: tt_lib.fallback_ops.fallback_ops.silu

.. autofunction:: tt_lib.fallback_ops.fallback_ops.softmax

.. autoclass:: tt_lib.fallback_ops.fallback_ops.Conv2d

.. autoclass:: tt_lib.fallback_ops.fallback_ops.GroupNorm

.. autoclass:: tt_lib.fallback_ops.fallback_ops.LayerNorm

Experimental Operations
=======================

Operations in this section are experimental, don't have full support, and may behave in unexpectedx ways.

Fused Operations from ``tt_lib`` Mini-Graph Library
---------------------------------------------------

We have a variety of common operations that require fusion of multiple
base operations together.

.. autofunction:: tt_lib.fused_ops.linear.Linear

.. autofunction:: tt_lib.fused_ops.softmax.softmax

.. autofunction:: tt_lib.fused_ops.layernorm.Layernorm

.. autofunction:: tt_lib.fused_ops.add_and_norm.AddAndNorm

Other Operations
----------------

.. autofunction:: tt_lib.tensor.transpose_hc_rm

.. autofunction:: tt_lib.tensor.tilize

.. autofunction:: tt_lib.tensor.untilize

.. autofunction:: tt_lib.tensor.fill_rm

.. autofunction:: tt_lib.tensor.fill_ones_rm

.. autofunction:: tt_lib.tensor.pad_h_rm

.. autofunction:: tt_lib.tensor.large_bmm

.. autofunction:: tt_lib.tensor.large_bmm_single_block

.. autofunction:: tt_lib.tensor.conv

.. autofunction:: tt_lib.tensor.bert_large_fused_qkv_matmul

.. autofunction:: tt_lib.tensor.bert_large_ff1_matmul

.. autofunction:: tt_lib.tensor.bert_large_ff2_matmul

.. autofunction:: tt_lib.tensor.bert_large_selfout_matmul

.. autofunction:: tt_lib.tensor.bert_large_pre_softmax_bmm

.. autofunction:: tt_lib.tensor.bert_large_post_softmax_bmm

.. autofunction:: tt_lib.tensor.softmax

.. autofunction:: tt_lib.tensor.scale_mask_softmax

.. autofunction:: tt_lib.tensor.layernorm

.. autofunction:: tt_lib.tensor.layernorm_gamma

.. autofunction:: tt_lib.tensor.layernorm_gamma_beta

.. autofunction:: tt_lib.tensor.add_layernorm_gamma_beta

.. autofunction:: tt_lib.tensor.tilize_with_zero_padding

.. autofunction:: tt_lib.tensor.tilize_conv_activation

.. autofunction:: tt_lib.tensor.convert_conv_weight_tensor_to_tiled_layout
