

noc_semaphore_inc
=================

The Tensix core executing this function call initiates an atomic increment (with 32-bit wrap) of a remote Tensix core L1 memory address. This L1 memory address is used as a semaphore of size 4 Bytes, as a synchronization mechanism.

Return value: None

.. list-table::
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - addr
     - Encoding of the destination location (x,y)+address
     - uint64_t
     - TODO(insert a reference to what constitutes valid coords)
   * - incr
     - The value to increment by
     - uint32_t
     - Any `uint32_t` value
