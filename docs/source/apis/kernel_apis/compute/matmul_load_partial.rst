
matmul_load_partial
===================

Loads a submatrix element of a tile-sized matrix into register DST at a specified index for subsequent use with `matmul_tiles`.
The DST register buffer must be in acquired state via `acquire_dst` call.
This call is blocking and is only available on the compute engine.

TODO(AP): needs review/better description.

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - in_cb_id
     - The identifier of the source circular buffer (CB)
     - uint32_t
     - 0 to 31
   * - in_tile_index
     - The index of the tile to copy from the input CB
     - uint32_t
     - Must be less than the size of the CB
   * - dst_tile_index
     - The index of the tile in the DST register
     - uint32_t
     - Must be less than the size of the DST register (16)
