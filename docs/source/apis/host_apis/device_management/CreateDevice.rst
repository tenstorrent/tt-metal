CreateDevice()
===============

Instantiates a device object. 

Return value: Device *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - device_type
     - Type of Tenstorrent device to be used
     - ARCH enum
     - "tt::ARCH::GRAYSKULL"
   * - pcie_slot
     - The number of the PCIexpress slot in which the device is located
     - int
     - 0 to 7