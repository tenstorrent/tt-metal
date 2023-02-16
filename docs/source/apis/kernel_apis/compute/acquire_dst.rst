

acquire_dst
===========

Acquires an exclusive lock on the internal DST register for the current Tensix core. This register is an array of 16 tiles of 32x32 elements each.
If the lock is already acquired, this function will wait until it is released.

This call is blocking and is only available on the compute engine.

TODO(Describe meanings of dst_mode values).

Return value: None

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - core
     - TODO(Argument will be removed in the near future)
     - 
     - 
   * - dst_mode
     - Specifies how the destination register is going to be used
     - uint32_t
     - DstMode::Full, DstMode::Half, DstMode::Tile
