TT-DNN
******

Overview
========

TT-DNN is a simplified Python interface to the compute engine of the TT-Metal.

This will be the future plan. For now, the ``ttlib`` Python module is a
unified Python interface that provides both TT-DNN and the Tensor library.

tt-DNN API through ``ttlib``
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

.. autofunction:: ttlib.tensor.add

.. autofunction:: ttlib.tensor.sub

.. autofunction:: ttlib.tensor.mul

.. autofunction:: ttlib.tensor.matmul

.. autofunction:: ttlib.tensor.bmm

.. autofunction:: ttlib.tensor.exp

.. autofunction:: ttlib.tensor.recip

.. autofunction:: ttlib.tensor.gelu

.. autofunction:: ttlib.tensor.sqrt

.. autofunction:: ttlib.tensor.sigmoid

.. autofunction:: ttlib.tensor.log

.. autofunction:: ttlib.tensor.tanh

Tensor manipulation operations
------------------------------

These operations change the tensor shape in some way, giving it new dimensions
but in general retaining the data.

.. autofunction:: ttlib.tensor.reshape

.. autofunction:: ttlib.tensor.transpose

.. autofunction:: ttlib.tensor.transpose_hc

.. autofunction:: ttlib.tensor.transpose_hc_rm

.. autofunction:: ttlib.tensor.tilize

.. autofunction:: ttlib.tensor.untilize

All other operations
--------------------

.. autofunction:: ttlib.tensor.fill_rm

.. autofunction:: ttlib.tensor.fill_ones_rm

.. autofunction:: ttlib.tensor.pad_h_rm

.. autofunction:: ttlib.tensor.bcast

.. autofunction:: ttlib.tensor.reduce

Enums
-----

.. autoclass:: ttlib.tensor.BcastOpMath
    :members: ADD, SUB, MUL

.. autoclass:: ttlib.tensor.BcastOpDim
    :members: H, W, HW

.. autoclass:: ttlib.tensor.ReduceOpMath
    :members: SUM, MAX

.. autoclass:: ttlib.tensor.ReduceOpDim
    :members: H, W, HW

``ttlib`` Mini-Graph Library
==============================

Fused Operations
----------------

We have a variety of common operations that require fusion of multiple
base operations together.

.. autofunction:: ttlib.fused_ops.linear.Linear

.. autofunction:: ttlib.fused_ops.softmax.softmax

.. autofunction:: ttlib.fused_ops.layernorm.Layernorm

.. autofunction:: ttlib.fused_ops.add_and_norm.AddAndNorm
