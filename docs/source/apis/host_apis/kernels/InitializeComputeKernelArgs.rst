InitializeComputeKernelArgs()
=============================

Creates kernel arguments for compute kernel

Return value: ComputeKernelArgs *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - logical_core
     - The location of the Tensix core with a kernel that receives these arguments (Logical co-ordinates)
     - const tt_xy_pair & 
     - {0, 0} --> {9, 11}
   * - compile_time_args
     - A pointer to the struct containing the args. Struct definition is located in the \*.cpp file of the kernel
     - void *
     - 
   * - compile_time_args_size
     - Size of struct containing the kernel arguments
     - size_t
     - 0 to 512 Bytes


Creates the same kernel arguments for a range of cores

Return value: ComputeKernelArgs *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - core_range
     - The range of the Tensix co-ordinates with a kernel that receives these arguments (Logical co-ordinates)
     - const CoreRange & (std::pair<tt_xy_pair, tt_xy_pair>)
     - Any range encompassing cores within {0 , 0} --> {9, 11}
   * - compile_time_args
     - A pointer to the struct containing the args. Struct definition is located in the \*.cpp file of the kernel
     - void *
     - 
   * - compile_time_args_size
     - Size of struct containing the kernel arguments
     - size_t
     - 0 to 512 Bytes


Creates kernel arguments specified by a combination of single core co-ordinates or a range of core co-ordinates

Return value: ComputeKernelArgs *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - core_blocks
     - A collection containing a single Tensix co-ordinate or a range of Tensix co-ordinates that receives these arguments (Logical co-ordinates)
     - const CoreBlocks & (std::vector<std::variant<tt_xy_pair, CoreRange>>)
     - A single core or range encompassing cores within {0 , 0} --> {9, 11}
   * - compile_time_args
     - A collection of pointers to structs containing the args. Struct definition is located in the \*.cpp file of the kernel.
     - const std::vector<void ``*``> &
     - Same size as core_blocks. Args are assigned to core or range of cores from core_blocks in order of compile_time_args. 
   * - compile_time_args_size
     - Size of struct containing the kernel arguments
     - size_t
     - 0 to 512 Bytes
