ttmetal Python API (will become ttDNN)
**************************************

The TT-Metal Tensor library provides easy entry to the data structures and
device management layer of TT-Metal.

ttDNN is a simplified Python interface to the compute engine of the TT-Metal.

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

.. autofunction:: ttmetal.device.CreateDevice

.. autofunction:: ttmetal.device.InitializeDevice

.. autofunction:: ttmetal.device.GetHost

.. autofunction:: ttmetal.device.CloseDevice

.. autoclass:: ttmetal.tensor.Tensor
    :members:
    :special-members: __init__

.. autofunction:: ttmetal.utils.pad_activation

.. autofunction:: ttmetal.utils.pad_weight

.. autofunction:: ttmetal.utils.tilize_to_list

.. autofunction:: ttmetal.utils.untilize

Tensor Examples with ``torch``
------------------------------

Tensors in TT-Metal only work if they are

* 4-dimensional
* Their final two dimensions (height and width) are divisible by 32

Furthermore, before they're passed to the device and its compute engine, they
need to be tilized.

We have some examples of using TT-Metal tensors with PyTorch.

Converting a PyTorch Tensor to a TT-Metal Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    ...
    import ttmetal

    pcie_0 = 0
    device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, pcie_0)

    pt_tensor = torch.randn(1, 1, 5, 128)
    padded_pt_tensor = ttmetal.utils.pad_activation(pt_tensor)

    tilized_pt_tensor = ttmetal.utils.tilize_to_list(padded_pt_tensor)

    tt_shape = list(padded_pt_tensor.shape())
    tt_tensor = ttmetal.tensor.Tensor(
        tilized_pt_tensor,
        tt_shape,
        ttmetal.tensor.DataType.BFLOAT16,
        ttmetal.tensor.Layout.TILE,
        device
    )

First, this will pad a 4D-tensor called ``embeddings`` to a shape valid for
TT-Metal. These will how you will be changing the shape of your activations
before passing them in.

Next, you tilize the padded tensor, resulting in a list of values. It's now fit
for final conversion.

Finally, create a real TT-Metal tensor by providing the tilized, padded tensor
value list with some options for tensor.

Converting a TT-Metal Tensor back to PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import ttmetal

    ...

    pcie_0 = 0
    device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, pcie_0)
    host = ttmetal.device.GetHost()

    shape = tt_tensor.shape()

    tt_tensor_out = tt_tensor.to(host)

    tt_out_tilized = torch.Tensor(tt_tensor_out.data())
    tt_out_flat = ttmetal.utils.untilize(tt_out_tilized)
    tt_out = tt_out_flat.reshape(shape)

Converting back is even easier.

First, with a valid ``Device``, get the ``Host``.

Then, transfer the TT-Metal tensor back to host memory space.

Finally, create a flat PyTorch tensor, untilize it, and reshape it back into
your desired shape.

Remember that this is padded, so you'll have to manually erase any padding
that padding calls did.

ttDNN through ``ttmetal``
=========================

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
