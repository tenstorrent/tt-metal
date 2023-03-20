ttDNN
*****

Overview
========

ttDNN is a simplified Python interface to the compute engine of the TT-Metal.

This will be the future plan. For now, the ``ttmetal`` Python module is a
unified Python interface that provides both ttDNN and the Tensor library.

ttDNN API through ``ttmetal``
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

.. autofunction:: ttmetal.tensor.add

.. autofunction:: ttmetal.tensor.sub

.. autofunction:: ttmetal.tensor.mul

.. autofunction:: ttmetal.tensor.matmul

.. autofunction:: ttmetal.tensor.bmm

.. autofunction:: ttmetal.tensor.exp

.. autofunction:: ttmetal.tensor.recip

.. autofunction:: ttmetal.tensor.gelu

.. autofunction:: ttmetal.tensor.sqrt

.. autofunction:: ttmetal.tensor.sigmoid

.. autofunction:: ttmetal.tensor.log

.. autofunction:: ttmetal.tensor.tanh

Tensor manipulation operations
------------------------------

These operations change the tensor shape in some way, giving it new dimensions
but in general retaining the data.

.. autofunction:: ttmetal.tensor.reshape

.. autofunction:: ttmetal.tensor.transpose

.. autofunction:: ttmetal.tensor.transpose_hc

.. autofunction:: ttmetal.tensor.transpose_hc_rm

.. autofunction:: ttmetal.tensor.tilize

.. autofunction:: ttmetal.tensor.untilize

All other operations
--------------------

.. autofunction:: ttmetal.tensor.fill_rm

.. autofunction:: ttmetal.tensor.fill_ones_rm

.. autofunction:: ttmetal.tensor.pad_h_rm

.. autofunction:: ttmetal.tensor.bcast

.. autofunction:: ttmetal.tensor.reduce

Enums
-----

.. autoclass:: ttmetal.tensor.BcastOpMath
    :members: ADD, SUB, MUL

.. autoclass:: ttmetal.tensor.BcastOpDim
    :members: H, W, HW

.. autoclass:: ttmetal.tensor.ReduceOpMath
    :members: SUM, MAX

.. autoclass:: ttmetal.tensor.ReduceOpDim
    :members: H, W, HW

``ttmetal`` Mini-Graph Library
==============================

Fused Operations
----------------

We have a variety of common operations that require fusion of multiple
base operations together.

.. autofunction:: ttmetal.fused_ops.linear.Linear

.. autofunction:: ttmetal.fused_ops.softmax.softmax

.. autofunction:: ttmetal.fused_ops.layernorm.Layernorm

.. autofunction:: ttmetal.fused_ops.add_and_norm.AddAndNorm
