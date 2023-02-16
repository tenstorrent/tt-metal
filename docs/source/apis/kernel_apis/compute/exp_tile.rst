

exp_tile
========

Performs element-wise computation of exponential on each element of a tile in DST register at index tile_index.
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
   * - tile_index
     - The index of the tile in DST register buffer to perform the computation on
     - uint32_t
     - Must be less than the size of the DST register buffer
