CreateDataMovementKernel()
===========================

Creates a single core data movement kernel and adds it to the program. 

Return value: DataMovementKernel *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - program
     - The program to which this kernel will be added to
     - Program * 
     - 
   * - file_name
     - Name of file containing the kernel
     - const std::string
     - 
   * - core
     - The location of the Tensix core on which the kernel will execute (Logical co-ordinates)
     - const tt_xy_pair &
     - {0, 0} --> {9, 11}
   * - kernel_args
     - Compile and runtime kernel arguments passed at compile time and runtime respectively
     - DataMovementKernelArgs *
     - 
   * - processor_type
     - The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V)
     - enum
     - DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1
   * - noc
     - The NoC ID on which the kernel will perform data transfers 
     - enum
     - RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,

Creates a multi-core data movement kernel and adds it to the program. 

Return value: DataMovementKernel *

.. list-table:: 
   :widths: 25 50 25 25
   :header-rows: 1

   * - Argument
     - Description
     - Data type
     - Valid range
   * - program
     - The program to which this kernel will be added to
     - Program * 
     - 
   * - file_name
     - Name of file containing the kernel
     - const std::string
     - 
   * - core_range
     - The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)
     - const CoreRange &
     - Any range encompassing cores within {0 , 0} --> {9, 11}
   * - kernel_args
     - Compile and runtime kernel arguments passed at compile time and runtime respectively
     - DataMovementKernelArgs *
     - 
   * - processor_type
     - The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V)
     - enum
     - DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1
   * - noc
     - The NoC ID on which the kernel will perform data transfers 
     - enum
     - RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,

