.. _Tensor:

Tensor
******

Overview
========

The TT Tensor library provides support for creation and manipulation of TT Tensors.

This library is used by TT-Dispatch to represent tensors that can be sent to and received from TT-Metal platform.
Operations in ttDNN library also utilize this library: operation take TT Tensors as inputs and return TT Tensors as outputs.

This library only supports tensors of rank 4.

TT Tensor library provides support for different memory layouts of data stored within tensor.

ROW_MAJOR layout will store values in memory row by row, starting from last dimension of tensor.
For a tensor of shape ``[W, Z, Y, X]`` to be stored in ROW_MAJOR order on TT Accelerator device, ``X`` must be divisible by 2.
A tensor in ROW_MAJOR order with ``X`` not divisible by 2 can exist on host machine, but can't be sent TT Accelerator device.
So you can't provide a TT Accelerator device to TT Tensor construct for this type of tensor nor can you use ``ttnn.Tensor.to()``
to send this type of tensor to TT Accelerator device.

TILE layout will store values in memory tile by tile, starting from the last two dimensions of the tensor.
A tile is a (32, 32) shaped subsection of tensor.
Tiles are stored in memory in row major order, and then values inside tiles are stored in row major order.
A TT Tensor of shape ``[W, Z, Y, X]`` can have TILE layout only if both ``X`` and ``Y`` are divisible by 2.

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

Tensor Storage
==============

Tensor class has 3 types of storages: `OwnedStorage`, `BorrowedStorage` and `DeviceStorage`. And it has a constructor for each type.

`OwnedStorage` is used to store the data in host DRAM. Every data type is stored in the vector corresponding to that data type.
And the vector itself is stored in the shared pointer. That is done so that if the Tensor object is copied, the underlying storage is simply reference counted and not copied as well.

`BorrowedStorage` is used to borrow buffers from `torch`, `numpy`, etc

`DeviceStorage` is used to store the data in device DRAM or device L1. It also uses a shared pointer to store the underlying buffer. And the reason is also to allow for copying Tensor objects without having to copy the underlying storage.

Tensor API
==========

.. autoclass:: ttnn.Tensor
    :members: to, buffer, storage_type, device, get_layout, get_dtype, pad, unpad, pad_to_tile, unpad_from_tile
    :special-members: __init__

MemoryConfig
============
.. autoclass:: ttnn.MemoryConfig
    :special-members: __init__

Examples of converting between PyTorch Tensor and TT Tensor
===========================================================

Remember that TT Tensors must have rank 4.

Converting a PyTorch Tensor to a TT Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to create a TT Tensor ``tt_tensor`` from a PyTorch tensor.
The created tensor will be in ROW_MAJOR layout and stored on TT accelerator device.

.. code-block:: python

    py_tensor = torch.randn((1, 1, 32, 32))
    tt_tensor = ttnn.Tensor(py_tensor, ttnn.bfloat16).to(tt_device)

Converting a TT Tensor to a PyTorch Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to move a TT Tensor ``output`` from device to host and how to convert a TT Tensor to PyTorch tensor.

.. code-block:: python

    # move TT Tensor output from TT accelerator device to host
    tt_output = tt_output.cpu()

    # create PyTorch tensor from TT Tensor using to_torch() member function
    py_output = tt_output.to_torch()
