.. _Tilized_data_layout:

Tilized Data Layout
===================

The tilized data layout is a commonly used in Tenstorrent kernels. It is a
layout that moves nearby elements in a tensor's 2D plane to nearby memory
locations and is often efficient for compute operations and data-movement.

Where as in row-major, the next element down or up a row in the tensor may be
quite far in memory, depending on the width of the tensor, in the tilized
data-layout, those elements will be relatively close if they are in the same
tile.

Tilization
----------

Tilization is the transformation to take a tensor into a tilized data-layout.
The process can be interpreted as applying a tumbling window (2D) of the
tensor.

Below is an example where we are tilizing a 4x8 row-major tensor with 2x2 sized
tiles. The contents of each element show the element offset if counting in
row-major layout. The left column shows the offset of the row in memory and
assumes 1B sized elements. In the tilized example below, the tumbling window is
a row-major scan of the tensor.

::

    # Untilized (row-major) layout
          |---|---|---|---|---|---|---|---|
     0x0  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
          |---|---|---|---|---|---|---|---|
     0x8  | 8 | 9 | 10| 11| 12| 13| 14| 15|
          |---|---|---|---|---|---|---|---|
     0x10 | 16| 17| 18| 19| 20| 21| 22| 23|
          |---|---|---|---|---|---|---|---|
     0x18 | 24| 25| 26| 27| 28| 29| 30| 31|
          |---|---|---|---|---|---|---|---|

    # Tilized (2x2 tile) layout.
          |---|---|       |---|---|       |---|---|       |---|---|
     0x0  | 0 | 1 |   0x4 | 2 | 3 |   0x8 | 4 | 5 |   0xC | 6 | 7 |
          |---|---|       |---|---|       |---|---|       |---|---|
     0x2  | 8 | 9 |   0x6 | 10| 11|   0xA | 12| 13|   0xE | 14| 15|
          |---|---|       |---|---|       |---|---|       |---|---|

          |---|---|       |---|---|       |---|---|       |---|---|
     0x10 | 16| 17|  0x14 | 18| 19|  0x18 | 20| 21|  0x1C | 22| 23|
          |---|---|       |---|---|       |---|---|       |---|---|
     0x12 | 24| 25|  0x16 | 26| 27|  0x1A | 28| 29|  0x1E | 30| 31|
          |---|---|       |---|---|       |---|---|       |---|---|
