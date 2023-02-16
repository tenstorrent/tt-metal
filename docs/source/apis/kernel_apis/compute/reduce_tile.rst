

reduce_tile
===========

Performs a reduction operation `B = reduce(A)` using reduce_func for dimension reduction on a tile in the CB at a given index and writes the result to the DST register at index dst_tile_index.
Reduction can be either of type Reduce::R, Reduce::C or Reduce::RC, identifying the dimension(s) to be reduced in size to 1.
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
   * - reduce_func
     - Enum value, specifying the type of reduce function to perform.
     - uint32_t
     - One of ReduceFunc::Sum, ReduceFunc::Max
   * - dim
     - Dimension id, identifying the dimension to reduce in size to 1.
     - uint32_t
     - One of Reduce::R, Reduce::C, Reduce::RC
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
   * - coeff
     - Scaling factor applied to each element of the resulting tile.
     - float
     - any float number

