

cb_reserve_back
===============

A blocking call that waits for the specified number of tiles to be free in the specified circular buffer.
This call is used by the producer to wait for the consumer to consume (ie. free up) the specified number of tiles.

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
     - The number of free tiles to wait for
     - uint32_t
     - It must be less or equal than the size of the CB (the total number of tiles that fit into the CB)
