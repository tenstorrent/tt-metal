

copy_tile
=========

Copies a single tile from the specified input CB and writes the result to DST at a specified index.
For the `in_tile_index` to be valid for this call, `cb_wait_front(n)` had to be previously called to ensure that at least some number n\>0 of tiles are available in the input CB.
The CB index 0 then references the first tile in the received section of the CB, up to index n-1 (in a FIFO order).
The DST register buffer must be in acquired state via `acquire_dst` call. This call is blocking and is only available on the compute engine.

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
