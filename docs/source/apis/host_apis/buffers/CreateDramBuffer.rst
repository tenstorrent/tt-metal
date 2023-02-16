CreateDramBuffer()
===========================

Creates a DRAM Buffer object. DRAM buffers exist independently from a program

Return value: Buffer *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - dram_channel
     - DRAM channel ID
     - int
     - [0, 7]
   * - size_in_bytes
     - Size of DRAM buffer in Bytes
     - uint32_t
     - TODO: valid range?  0 to 2 GB (expressed in Bytes)
   * - address
     - Address at which the DRAM buffer will reside
     - uint32_t
     - TODO: fix range.  0 to 2 GB??
