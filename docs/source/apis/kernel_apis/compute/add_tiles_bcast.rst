

add_tiles_bcast
===============

This document applies to either one of the 3 broadcast operation variants - `add_tiles_bcast`, `sub_tiles_bcast` and `mul_tiles_bcast`.

The description below describes add_tiles_bcast, the other 2 operations use the same definition with the corresponding substitution of the math operator.

Performs a broadcast-operation `C=A\+B` of tiles in two CBs at given indices and writes the result to the DST register at index dst_tile_index.
The DST register buffer must be in acquired state via `acquire_dst` call.
This call is blocking and is only available on the compute engine.

Broadcasting semantics are defined as follows:

For dim==Dim::R, the input in `B` is expected to be a single tile with a filled 0-column and zeros elsewhere. |br|
The result is `C[h, w] = A[h,w] + B[w]`


For dim==Dim::C, the input in `B` is expected to be a single tile with a filled 0-row, and zeros elsewhere. |br|
The result is `C[h, w] = A[h,w] + B[h]`

For dim==Dim::RC, the input in `B` is expected to be a single tile with a filled single value at location [0,0], and zeros elsewhere. |br|
The result is `C[h, w] = A[h,w] + B[0,0]`

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - dim
     - Broadcast dimension
     - uint32_t
     - One of Dim::R, Dim::C, Dim::RC.
   * - in0_cb_id
     - The identifier of the circular buffer (CB) containing A
     - uint32_t
     - 0 to 31
   * - in1_cb_id
     - The indentifier of the circular buffer (CB) containing B
     - uint32_t
     - 0 to 31
   * - in0_tile_index
     - The index of tile A within the first CB
     - uint32_t
     - Must be less than the size of the CB
   * - in1_tile_index
     - The index of tile B within the second CB
     - uint32_t
     - Must be less than the size of the CB
   * - dst_tile_index
     - The index of the tile in DST REG for the result C
     - uint32_t
     - Must be less than the acquired size of DST REG

TODO(AP): verify that the bcast tile is actually required to be filled with zeros.


.. |br| raw:: html

      <br>

