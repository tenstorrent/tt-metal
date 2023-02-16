CreateL1Buffer()
===========================

Creates an L1 buffer and adds it to the program. 

Return value: Buffer * 

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
   * - size_in_bytes
     - Size of DRAM buffer in Bytes
     - uint32_t
     - TODO: valid range?  0 to 800 KB ?? (expressed in Bytes)
   * - address
     - Address at which the DRAM buffer will reside
     - uint32_t
     - TODO: fix range.  200KB to 1MB ??? (expressed in Bytes)
