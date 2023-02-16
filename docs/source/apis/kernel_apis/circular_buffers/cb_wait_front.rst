

cb_wait_front
=============

A blocking call that waits for the specified number of tiles to be available in the specified circular buffer (CB).
This call is used by the consumer of the CB to wait for the producer to fill the CB with at least the specfied number of tiles.

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
     - The number of tiles to wait for
     - uint32_t
     - It must be less or equal than the size of the CB (the total number of tiles that fit into the CB)
