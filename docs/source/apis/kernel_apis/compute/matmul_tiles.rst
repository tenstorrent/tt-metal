

matmul_tiles
============

Performs tile-sized matrix multiplication C=A\*B between the tiles in two specified input CBs and writes the result to DST.
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
   * - in0_cb_id
     - The identifier of the first input circular buffer (CB)
     - uint32_t
     - 0 to 31
   * - in1_cb_id
     - The indentifier of the second input circular buffer (CB)
     - uint32_t
     - 0 to 31
   * - in0_tile_index
     - The index of the tile A from the first input CB
     - uint32_t
     - Must be less than the size of the CB
   * - in1_tile_index
     - The index of the tile B from the second input CB
     - uint32_t
     - Must be less than the size of the CB
   * - dst_tile_index
     - The index of the tile in DST REG to which the result C will be written. 
     - uint32_t
     - Must be less than the acquired size of DST REG
