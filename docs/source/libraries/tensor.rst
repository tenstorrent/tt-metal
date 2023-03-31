Tensor
******

Overview
========

The TT Tensor library provides support for creation and manipulation of TT Tensors.

This library is used by TT-Dispatch to represent tensors that can be sent to and received from TT-Metal platform.
Operations in ttDNN library also utilize this library: operation take TT Tensors as inputs and return TT Tensors as outputs.

This library only supports tensors of rank 4, where the sizes of last two dimensions must be both multiple of 32.


TT Tensor library provides support for different memory layouts of data stored within tensor.

ROW_MAJOR layout will store values in memory row by row, starting from last dimension of tensor.

TILE layout will store values in memory tile by tile, starting from the last two dimensions of the tensor.
A tile is a (32, 32) shaped subsection of tensor.
Tiles are stored in memory in row major order, and then values inside tiles are stored in row major order.

.. code-block::

    #Tensor of shape (2, 64, 64)

    #batch=0
    [    0,    1,    2, ...,   63,
        64,   65,   66, ...,  127,
        ...
      3968, 3969, 3970, ..., 4031,
      4032, 4033, 4034, ..., 4095 ]

    #batch=1
    [ 4096, 4097, 4098, ..., 4159,
      4160, 4161, 6462, ..., 4223,
        ...
      8064, 8065, 8066, ..., 8127,
      8128, 8129, 8130, ..., 8191 ]


    #Stored in ROW_MAJOR layout
    [0, 1, 2, ..., 63, 64, ..., 4095, 4096, 4097, 4098, ..., 4159, 4160, ..., 8191]

    #Stored in TILE layout
    [  0,    1, ...,   31,   64,   65, ...,   95, ..., 1984, 1985, ..., 2015, # first tile of batch=0
      32,   33, ...,   63,   96,   97, ...,  127, ..., 2016, 2017, ..., 2047, # second tile of batch=0
    ...
    2080, 2081, ..., 2111, 2144, 2145, ..., 2175, ..., 4064, 4065, ..., 4095, # fourth (last) tile of batch=0

    4096, ..., 6111,                                                           # first tile of batch=1
    ...
    6176, ..., 8191 ]                                                          # fourth (last) tile of batch=0

Tensor API
==========

.. autoclass:: tt_lib.tensor.Tensor
    :members: to, data, layout, print, pretty_print, shape
    :special-members: __init__

MemoryConfig
============
.. autoclass:: tt_lib.tensor.MemoryConfig
    :special-members: __init__

Examples of converting between PyTorch Tensor and TT Tensor
===========================================================

Remember that TT Tensors must

* have rank 4
* have their final two dimensions (height and width) divisible by 32

Converting a PyTorch Tensor to a TT Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to create a TT Tensor from a PyTorch tensor.
After creating TT Tensor, this example also shows how to set memory layout of TT Tensor to TILE and send the tensor to TT accelerator device.

.. code-block:: python

    py_tensor = torch.randn((1, 1, 32, 32))
    tt_tensor = (
        tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(), # PyTorch tensor flatten into a list of floats
            py_tensor.size(),               # shape of TT Tensor that will be created
            tt_lib.tensor.DataType.BFLOAT16, # data type that will be used in created TT Tensor
            tt_lib.tensor.Layout.ROW_MAJOR,  # memory layout that will be used in created TT Tensor
        )
        .to(tt_lib.tensor.Layout.TILE)     # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)                         # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )

Converting a TT Tensor to a PyTorch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to move a TT Tensor output from device to host and how to convert a TT Tensor to PyTorch tensor.

.. code-block:: python

    # move TT Tensor output from TT accelerator device to host
    # and then on host, change memory layout of TT Tensor to ROW_MAJOR
    tt_output = output.to(host).to(tt_lib.tensor.Layout.ROW_MAJOR)

    # create a 1D PyTorch tensor from values in TT Tensor obtained with data() member function
    # and then reshape PyTorch tensor to shape of TT Tensor
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
