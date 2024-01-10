Tensor
=======

.. autoclass:: ttnn.Tensor


A `ttnn.Tensor` is a multi-dimensional matrix containing elements of a single data type.

Shape
*****

`ttnn.Tensor` uses `ttnn.Shape` to store its shape.

A shape of `ttnn.Tensor` can be obtained by using `shape` property of `ttnn.Tensor`.

`ttnn.Shape` is n-dimensional and can have the rank from 1 to 8.

`ttnn.Shape` stores the shape of the actual data and the shape of the padded data which are different due to hardware requirements.

`ttnn.Shape([16, 32])` creates a 2D tensor with 16 rows and 32 columns. Where the number of actual rows and columns and the number of padded rows and columns is 16 and 32 respectively. The tensor of this shape would have 16 * 32 elements in the storage.

`ttnn.Shape([14, 31], (32, 32))` creates a 2D tensor with 14 rows and 31 columns. Where the number of actual rows and columns is 14 and 31 respectively and the number of padded rows and columns is 32 and 32 respectively. The tensor of this shape would have 32 * 32 elements in the storage.
Printing the shape would show the actual shape and the padded shape as follows:

.. code-block:: python

    >>> print(ttnn.Shape([14, 31], (32, 32)))
    ttnn.Shape([14 + 18, 31 + 1])


Padded shape can be obtained by calling `padded()` method of `ttnn.Shape`

.. code-block:: python

    >>> print(ttnn.Shape([14, 31], (32, 32)).padded())
    ttnn.Shape([32, 32])


Layout
******

Data Type
*********

Storage
*******

Memory Config
*************

Gotchas
*******
Even size on width needed and why
