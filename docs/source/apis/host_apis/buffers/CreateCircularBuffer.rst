CreateCircularBuffer()
===========================

Creates a Circular Buffer (CB) in L1 memory and adds it to the program. 

Return value: CircularBuffer * 

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - program
     - The program to which buffer will be added to.
     - Program * 
     - 
   * - buffer_index
     - The index/ID of the CB. 
     - uint32_t
     - 0 to 32 TODO: specify more detail here.
   * - core
     - The location of the Tensix core on which the CB will reside (SoC co-ordinates)
     - const tt_xy_pair &
     - TODO: { , } --> { , }
   * - num_tiles
     - Total number of tiles to be stored in the CB
     - uint32_t
     - TODO: range?
   * - size_in_bytes
     - Size of CB buffer in Bytes
     - uint32_t
     - 0 to 1 MB (TODO: in Bytes)
   * - l1_address
     - Address at which the CB buffer will reside
     - uint32_t
     - 200 kB to 1MB (TODO: in bytes)
   * - data_format
     - The format of the data to be stored in the CB
     - DataFormat enum
     - DataFormat::Float16_b
