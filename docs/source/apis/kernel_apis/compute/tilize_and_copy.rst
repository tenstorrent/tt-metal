

tilize_and_copy
===============

Converts the input tile from a row-major format to a 4-faces row-major format and copies to the DST register at specified index.
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
   * - num_tiles_c
     - TODO(AP): need to ask what this does.
     - uint32_t
     - TODO(AP)

TODO(AP): this is a guess at this point, needs a review of correctness.
TODO(AP): needs a definition of 4-faces row-major format.

