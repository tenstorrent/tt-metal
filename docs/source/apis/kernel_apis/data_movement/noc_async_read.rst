

noc_async_read
==============

Initiates an asynchronous read from a specified source node located at NOC coordinates (x,y) at a local address (encoded as a uint64_t using `get_noc_addr` function).
The destination is in L1 memory on the Tensix core executing this function call.  Also, see `noc_async_read_barrier`.

The source node can be either a DRAM bank, a Tensix core or a PCIe controller.

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - src_noc_addr
     - Encoding of the source DRAM location (x,y)+address
     - uint64_t
     - TODO(insert a reference to what constitutes valid coords)
   * - dst_local_l1_addr
     - Address in local L1 memory
     - uint32_t
     - 0..1MB
   * - size
     - Size of data transfer in bytes
     - uint32_t
     - 0..1MB
