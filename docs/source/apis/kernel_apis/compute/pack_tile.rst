

pack_tile
=========

Copies a single tile from the DST register buffer at a specified index to a specified CB at a given index.
For the `out_tile_index` to be valid for this call, `cb_reserve_back(n)` had to be called first to reserve at least some number n\>0 of tiles in the output CB.
The `out_tile_index` = 0 then references the first tile in the reserved section of the CB, up to index n-1 that will then be visible to the consumer in the same order after a cb_push_back call.
The DST register buffer must be in acquired state via `acquire_dst` call.
This call is blocking and is only available on the compute engine.


Operates in tandem with functions `cb_reserve_back` and `cb_push_back`.

A typical use case is first the producer ensures that there is a number of tiles available in the buffer via `cb_reserve_back`, then the producer uses the `pack_tile` call to copy a tile from one of DST slots to a slot in reserved space and finally `cb_push_back` is called to announce visibility of the reserved section of the circular buffer to the consumer.

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - src_tile_index
     - The index of the tile in the DST register
     - uint32_t
     - Must be less than the size of the DST register (16)
   * - out_cb_id
     - The identifier of the output circular buffer (CB)
     - uint32_t
     - 0 to 31
   * - out_tile_index
     - The index of the tile in the output CB to copy to
     - uint32_t
     - Must be less than the size of the CB
