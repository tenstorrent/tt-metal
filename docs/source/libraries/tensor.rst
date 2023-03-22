Tensor
******

Overview
========

The TT-Metal Tensor library provides easy entry to the data structures and
device management layer of TT-Metal.

This will be the future plan. For now, the ``ttlib`` Python module is a
unified Python interface that provides both TT-DNN and the Tensor library.

Tensor API through ``ttlib``
==============================

.. autoclass:: ttlib.device.Arch
    :members: GRAYSKULL

.. autoclass:: ttlib.device.Device
    :members:

.. autoclass:: ttlib.device.Host
    :members:
    :special-members: __init__

.. autofunction:: ttlib.device.CreateDevice

.. autofunction:: ttlib.device.InitializeDevice

.. autofunction:: ttlib.device.GetHost

.. autofunction:: ttlib.device.CloseDevice

.. autoclass:: ttlib.tensor.Tensor
    :members:
    :special-members: __init__

.. autofunction:: ttlib.utils.pad_activation

.. autofunction:: ttlib.utils.pad_weight

.. autofunction:: ttlib.utils.tilize_to_list

.. autofunction:: ttlib.utils.untilize

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
    import ttlib

    pcie_0 = 0
    device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, pcie_0)

    pt_tensor = torch.randn(1, 1, 5, 128)
    padded_pt_tensor = ttlib.utils.pad_activation(pt_tensor)

    tilized_pt_tensor = ttlib.utils.tilize_to_list(padded_pt_tensor)

    tt_shape = list(padded_pt_tensor.shape())
    tt_tensor = ttlib.tensor.Tensor(
        tilized_pt_tensor,
        tt_shape,
        ttlib.tensor.DataType.BFLOAT16,
        ttlib.tensor.Layout.TILE,
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
    import ttlib

    ...

    pcie_0 = 0
    device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, pcie_0)
    host = ttlib.device.GetHost()

    shape = tt_tensor.shape()

    tt_tensor_out = tt_tensor.to(host)

    tt_out_tilized = torch.Tensor(tt_tensor_out.data())
    tt_out_flat = ttlib.utils.untilize(tt_out_tilized)
    tt_out = tt_out_flat.reshape(shape)

Converting back is even easier.

First, with a valid ``Device``, get the ``Host``.

Then, transfer the TT-Metal tensor back to host memory space.

Finally, create a flat PyTorch tensor, untilize it, and reshape it back into
your desired shape.

Remember that this is padded, so you'll have to manually erase any padding
that padding calls did.
