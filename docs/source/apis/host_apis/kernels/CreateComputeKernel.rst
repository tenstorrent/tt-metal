CreateComputeKernel()
===========================

Creates a single core compute kernel object, and adds it to the program. 

Return value: ComputeKernel *

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
     - Kernel arguments, passed at compile time
     - ComputeKernelArgs *
     - 
   * - math_fidelity
     - The percision of the matrix compute engine
     - enum
     - MathFidelity::HiFi4
   * - fp32_dest_acc_en
     - Specifies the type of accumulation performed in the matrix compute engine. 
     - bool 
     - false (for Grayskull)
   * - math_approx_mode
     - Used by the vector compute engine. (will be depricated)
     - bool 
     - true, false

Creates a multi-core compute kernel object, and adds it to the program. 

Return value: ComputeKernel *

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
     - Kernel arguments, passed at compile time
     - ComputeKernelArgs *
     - 
   * - math_fidelity
     - The percision of the matrix compute engine
     - enum
     - MathFidelity::HiFi4
   * - fp32_dest_acc_en
     - Specifies the type of accumulation performed in the matrix compute engine. 
     - bool 
     - false (for Grayskull)
   * - math_approx_mode
     - Used by the vector compute engine. (will be depricated)
     - bool 
     - true, false
