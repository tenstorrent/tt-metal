Tensor
=======

.. autoclass:: ttnn.Tensor


A :class:`ttnn.Tensor` is a multi-dimensional matrix containing elements of a single data type.

Shape
*****

.. autoclass:: ttnn.Shape

:class:`ttnn.Tensor` uses :class:`ttnn.Shape` to store its shape.

A shape of :class:`ttnn.Tensor` can be obtained by using `shape` property of `ttnn.Tensor`.

:class:`ttnn.Shape` is n-dimensional and can have the rank from 1 to 8.

:class:`ttnn.Shape` stores the shape of the actual data and the shape of the padded data which are different due to hardware requirements.

:class:`ttnn.Shape([16, 32])` creates a 2D tensor with 16 rows and 32 columns. Where the number of actual rows and columns and the number of padded rows and columns is 16 and 32 respectively. The tensor of this shape would have 16 * 32 elements in the storage.

:class:`ttnn.Shape([14, 31], (32, 32))` creates a 2D tensor with 14 rows and 31 columns. Where the number of actual rows and columns is 14 and 31 respectively and the number of padded rows and columns is 32 and 32 respectively. The tensor of this shape would have 32 * 32 elements in the storage.
Printing the shape would show the actual shape and the padded shape as follows:

.. code-block:: python

    >>> print(ttnn.Shape([14, 31], (32, 32)))
    ttnn.Shape([14 + 18, 31 + 1])


Padded shape can be obtained by calling `padded()` method of `ttnn.Shape`

.. code-block:: python

    >>> print(ttnn.Shape([14, 31], (32, 32)).padded())
    ttnn.Shape([32, 32])

.. _ttnn.Layout:

Layout
******

.. _ttnn.ROW_MAJOR_LAYOUT:

**ttnn.ROW_MAJOR_LAYOUT**
Defines the layout in row major where the elements are alligned such that the next element in the height direction is the distance of the width.

.. _ttnn.TILE_LAYOUT:

**ttnn.TILE_LAYOUT**
Defines the layout where the elements themselves are placed within a 32 by 32 square called a tile.  In order to transition to TILE_LAYOUT
where the tensors height or width are not divisible by 32, padding is automatically provided by :ref:`ttnn.to_layout<ttnn.to_layout>`.


.. _ttnn.DataType:

Data Type
*********
**uint32**
    DataType.UINT32
**float32**
    DataType.FLOAT32
**bfloat16**
    DataType.BFLOAT16
**bfloat8_b**
    DataType.BFLOAT8_B


.. _ttnn.Storage:

Storage
*******

**OWNED**

    The memory for the tensor is on the host and belongs only to the one Tensor

**BORROWED**

    The memory for the tensor is on the host and is being shared by other Tensors

**DEVICE**

    The memory for the tensor is allocated to the deivce.


.. _ttnn.MemoryConfig:

Memory Config
*************
**ttnn.DRAM_MEMORY_CONFIG**

    Defines the memory to be interleaved in shared memory.

**ttnn.L1_MEMORY_CONFIG**

    Defines the memory to be interleaved and available in the cache local to the core.

Gotchas
*******
Even size on width needed and why
