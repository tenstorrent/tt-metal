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

Tensor math operations
----------------------

All arguments to operations on tensors are tensors, regardless of arity of
operation (unary, binary etc.).

+----------+----------------------+-----------+-------------+----------+
| Argument | Description          | Data type | Valid range | Required |
+==========+======================+===========+=============+==========+
| arg      | Tensor argument      | Tensor    |             | Yes      |
+----------+----------------------+-----------+-------------+----------+

Any arguments which are exceptions to this rule will be noted in that
operation's listing.

.. autofunction:: tt_lib.tensor.add

.. autofunction:: tt_lib.tensor.sub

.. autofunction:: tt_lib.tensor.mul

.. autofunction:: tt_lib.tensor.matmul

.. autofunction:: tt_lib.tensor.bmm

.. autofunction:: tt_lib.tensor.exp

.. autofunction:: tt_lib.tensor.recip

.. autofunction:: tt_lib.tensor.gelu

.. autofunction:: tt_lib.tensor.sqrt

.. autofunction:: tt_lib.tensor.sigmoid

.. autofunction:: tt_lib.tensor.log

.. autofunction:: tt_lib.tensor.tanh

Tensor manipulation operations
------------------------------

These operations change the tensor shape in some way, giving it new dimensions
but in general retaining the data.

.. autofunction:: tt_lib.tensor.reshape

.. autofunction:: tt_lib.tensor.transpose

.. autofunction:: tt_lib.tensor.transpose_hc

.. autofunction:: tt_lib.tensor.transpose_hc_rm

.. autofunction:: tt_lib.tensor.transpose_cn

.. autofunction:: tt_lib.tensor.tilize

.. autofunction:: tt_lib.tensor.untilize

All other operations
--------------------

.. autofunction:: tt_lib.tensor.fill_rm

.. autofunction:: tt_lib.tensor.fill_ones_rm

.. autofunction:: tt_lib.tensor.pad_h_rm

.. autofunction:: tt_lib.tensor.bcast

.. autofunction:: tt_lib.tensor.reduce

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

``tt_lib`` Mini-Graph Library
==============================

Fused Operations
----------------

We have a variety of common operations that require fusion of multiple
base operations together.

.. autofunction:: tt_lib.fused_ops.linear.Linear

.. autofunction:: tt_lib.fused_ops.softmax.softmax

.. autofunction:: tt_lib.fused_ops.layernorm.Layernorm

.. autofunction:: tt_lib.fused_ops.add_and_norm.AddAndNorm
