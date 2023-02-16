

noc_async_write
===============

Initiates an asynchronous write from a source address in L1 memory on the Tensix core executing this function call.
The destination is specified using a uint64_t encoding referencing an on-chip node located at NOC coordinates (x,y) and a local address created using `get_noc_addr` function.
Also, see `noc_async_write_barrier`.

The destination node can be either a DRAM bank, Tensix core+L1 memory address or a PCIe controller.

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - src_local_l1_addr
     - Source address in local L1 memory
     - uint32_t
     - 0..1MB
   * - dst_noc_addr
     - Encoding of the destination DRAM location (x,y)+address
     - uint64_t
     - TODO(insert a reference to what constitutes valid coords)
   * - size
     - Size of data transfer in bytes
     - uint32_t
     - 0..1MB
