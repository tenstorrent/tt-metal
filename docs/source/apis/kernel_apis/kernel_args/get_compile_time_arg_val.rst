get_compile_time_arg_val
========================

Returns the value of a constexpr argument from `kernel_compile_time_args` array provided during kernel creation using `CreateDataMovementKernel`, `CreateComputeKernel` calls.

Return value: constexpr uint32_t

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
     - 0 to 31
