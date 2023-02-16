

cb_pop_front
============

Pops a specified number of tiles from the front of the specified CB. This also frees this number of tiles in the circular buffer.
This call is used by the consumer to free up the space in the CB.

We use the convention that the producer pushes tiles into the "back" of the CB queue and the consumer consumes tiles from the "front" of the CB queue.

Note that the act of reading of the tile data from the CB does not free up the space in the CB.
Waiting on available tiles and popping them is separated in order to allow the consumer to:
1) read the tile data from the CB via multiple reads of sub-tiles
2) access the tiles (or their sub-tiles) that are visible to the consumer by random access of the valid section of the CB

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - cb_id
     - The index of the cirular buffer (CB)
     - uint32_t
     - 0 to 31
   * - num_tiles
     - The number of tiles to be popped
     - uint32_t
     - It must be less or equal than the size of the CB (the total number of tiles that fit into the CB)
