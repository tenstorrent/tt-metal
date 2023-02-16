get_arg_val
===========

Returns the value of an argument from `kernel_args` array provided during kernel creation using `CreateDataMovementKernel`, `CreateComputeKernel` calls.

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - arg_idx
     - The index of the argument
     - uint32_t
     - 0 to 255
   * - T (template argument)
     - Data type of the returned argument
     - Any 4-byte sized type
     - N/A
