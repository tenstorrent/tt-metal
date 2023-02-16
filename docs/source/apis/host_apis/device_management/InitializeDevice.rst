
InitializeDevice()
===================

Initializes a device by creating a tt_cluster object and memory manager. Puts device into reset.

Currently device has a 1:1 mapping with tt_cluster and memory manager only allocates DRAM addresses.

Return value: bool

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - device
     - Pointer to a device object
     - Device *
     - 