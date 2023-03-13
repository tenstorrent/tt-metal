ttDNN Reference (currently ttmetal)
===================================

ttDNN is a simplified Python interface to the compute engine of the TT-Metal.

The TT-Metal Tensor library provides easy entry to the data structures and
device management layer of TT-Metal.

This will be the future plan. For now, the ``ttmetal`` Python module is a
unified Python interface that provides both ttDNN and the Tensor library.

Tensor Library through ``ttmetal``
==================================

.. autoclass:: ttmetal.device.Arch
    :members: GRAYSKULL

.. autoclass:: ttmetal.device.Device
    :members:

.. autoclass:: ttmetal.device.Host
    :members:
    :special-members: __init__

.. autoclass:: ttmetal.tensor.Tensor
    :members:
    :special-members: __init__

.. autofunction:: ttmetal.device.CreateDevice

.. autofunction:: ttmetal.device.InitializeDevice

.. autofunction:: ttmetal.device.CloseDevice

ttDNN through ``ttmetal``
=========================

Operations on tensors
---------------------

All arguments to operations on tensors are tensors, regardless of arity of
operation (unary, binary etc.).

+----------+----------------------+-----------+-------------+----------+
| Argument | Description          | Data type | Valid range | Required |
+==========+======================+===========+=============+==========+
| arg      | First tensor to add  | Tensor    |             | Yes      |
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
