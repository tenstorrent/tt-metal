

transpose_wh_tile
=================

Performs a 32x32 transpose operation `B[w,h] = A[h,w]` on a tile in the CB at a given index and writes the result to the DST register at index dst_tile_index.
The DST register buffer must be in acquired state via `acquire_dst` call.
This call is blocking and is only available on the compute engine.

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - in_cb_id
     - The identifier of the circular buffer (CB) containing A
     - uint32_t
     - 0 to 31
   * - in_tile_index
     - The index of tile A within the first CB
     - uint32_t
     - Must be less than the size of the CB
   * - dst_tile_index
     - The index of the tile in DST REG for the result B
     - uint32_t
     - Must be less than the acquired size of DST REG
