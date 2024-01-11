Tensor
######


A :class:`ttnn.Tensor` is a multi-dimensional matrix containing elements of a single data type.

Shape
*****

:class:`ttnn.Tensor` uses :class:`ttnn.Shape` to store its shape.

:class:`ttnn.Shape` can be used to store dimensions for a tensor of rank 1 to rank 8 (inclusive).

:class:`ttnn.Shape` stores the shape of both the actual data and the padded data. Which can be different due to hardware requirements.

:class:`ttnn.Shape([16, 32])` is a shape of 2D tensor with 16 rows and 32 columns. Where the number of actual rows and columns is 16 and 32 respectively.
And the padded dimensions match the actual dimensions. The tensor of this shape has 16 * 32 elements in the storage.

.. figure:: images/tensor.png
    :align: center
    :alt: Tensor

    Tensor with 16 rows and 32 columns.


Printing the shape would show the actual shape:

.. code-block:: python

    >>> print(ttnn.Shape([16, 32]))
    ttnn.Shape([16, 32])

:class:`ttnn.Shape([14, 28], [32, 32])` is a shape of 2D tensor with 14 rows and 28 columns.
Where the number of actual rows and columns is 14 and 28 respectively and the number of padded rows and columns is 32 and 32 respectively.
The tensor of this shape has 32 * 32 elements in the storage.

.. figure:: images/tensor_with_padded_shape.png
    :align: center
    :alt: Tensor With Padded Shape

    Tensor with 14 rows and 28 columns and padded to 32 rows and 32 columns.

Printing the shape would show the actual shape with the padding:

.. code-block:: python

    >>> print(ttnn.Shape([14, 28], [32, 32]))
    ttnn.Shape([14 + 18, 28 + 4])


Padded shape can be obtained by calling `padded()` method of `ttnn.Shape`

.. code-block:: python

    >>> print(ttnn.Shape([14, 28], [32, 32]).padded())
    ttnn.Shape([32, 32])

.. _ttnn.Layout:

Layout
******

.. _ttnn.ROW_MAJOR_LAYOUT:

**ttnn.ROW_MAJOR_LAYOUT**

Row major layout has the consecutive elements of a row next to each other.

.. figure:: images/tensor_with_row_major_layout.png
    :align: center
    :alt: Tensor With Row-Major Layout

    4x4 tensor with a row-major layout.

.. _ttnn.TILE_LAYOUT:

**ttnn.TILE_LAYOUT**

In tile layout, the elements themselves are placed within a 32x32 square called a tile.
The tiles themselves are then still stored in a row-major order. In order to transition to TILE_LAYOUT, :ref:`ttnn.to_layout<ttnn.to_layout>` can be used.
When the height or width of the tensor are not divisible by 32, padding is automatically provided.

.. figure:: images/tensor_with_tile_layout.png
    :align: center
    :alt: Tensor With Tile Layout

    4x4 tensor stored using 2x2 tiles. Note that ttnn Tensors can only have 32x32 tiles. This image is for illustrative purposes only.


.. _ttnn.DataType:

Data Type
*********

ttnn supports the following data types:

- **uint32**
- **float32**
- **bfloat16**
- **bfloat8_b**


.. note::
    :class:`ttnn.Tensor` uses a minimum of 4 bytes to store a row of the tensor on the device.
    That means that the minimum width of a tensor on the device is as follows:

    .. list-table:: Minimum width of a tensor on the device
        :widths: 25 25
        :header-rows: 1

        * - Data Type
          - Minimum Width
        * - ttnn.uint32
          - 1
        * - ttnn.float32
          - 1
        * - ttnn.bfloat16
          - 2
        * - ttnn.bfloat8_b
          - 32 (Special case because the tensor has to be in tile layout)


.. _ttnn.Storage:

Storage
*******

**OWNED**

    The buffer of the tensor is on the host and its allocation/deallocation is owned by ttnn.

**BORROWED**

    The buffer of the tensor is on the host and it was borrowed from ``torch`` / ``numpy`` / an external buffer.

**DEVICE**

    The buffer of the tensor is on a ttnn device.


.. _ttnn.MemoryConfig:

Memory Config
*************
**ttnn.DRAM_MEMORY_CONFIG**

    The buffer of the tensor is interleaved and is stored in DRAM.

**ttnn.L1_MEMORY_CONFIG**

    The buffer of the tensor is interleaved and is stored in the the local cache of a core


**Sharded Memory Configs**

    TODO: Add documentation for sharded memory configs

APIs
****

.. autoclass:: ttnn.Tensor
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :exclude-members: value

.. autoclass:: ttnn.Shape
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :exclude-members: value
