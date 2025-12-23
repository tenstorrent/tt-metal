Lab 1: Single Core Matrix Multiplication
========================================

Introduction
------------

Matrix multiplication is a fundamental operation in linear algebra with widespread applications in scientific computing, machine learning, and computational graphics.
The standard matrix multiplication algorithm transforms two input matrices into a third matrix using a simple computational procedure.

Basic Algorithm
---------------

Given two matrices A of shape (M, K) and B of shape (K, N), the resulting matrix C will have shape (M, N).
The classical matrix multiplication algorithm can be expressed using a triple-nested loop structure,
where each element of the resulting matrix is computed as the dot product of the corresponding row of A and the corresponding column of B:

.. code-block:: cpp

   for (int i = 0; i < M; i++) {
       for (int j = 0; j < N; j++) {
           C[i][j] = 0.0;
           for (int k = 0; k < K; k++) {
               C[i][j] += A[i][k] * B[k][j];
           }
       }
   }

Computational Complexity
------------------------

The naive implementation has a time complexity of O(M*N*K)


Row-Major Memory Layout
-----------------------

Memory space is a linear (one-dimensional) array of memory words.
However, we often need to represent data with more dimensions, such as two-dimensional matrices or n-dimensional tensors.
In C++, two-dimensional arrays are usually stored in memory using **row-major order**.
This means that elements of each row are stored contiguously in memory, with rows placed one after another.

Consider a 3×4 matrix::

   Matrix A:
   [a00  a01  a02  a03]
   [a10  a11  a12  a13]
   [a20  a21  a22  a23]

In row-major layout, this matrix is stored in memory as::

   Memory: [a00][a01][a02][a03][a10][a11][a12][a13][a20][a21][a22][a23]
           └───── row 0 ─────┘ └───── row 1 ─────┘  └───── row 2 ─────┘

When accessing element ``A[i][j]`` in a matrix with ``n`` columns, the memory address is calculated as::

   address = base_address + (i × n + j) × sizeof(element)

Cache-Friendly Access Patterns
-------------------------------

Modern CPUs use cache memory to reduce the latency of memory accesses.
When a memory location is accessed, the CPU loads an entire **cache line**, consisting of multiple consecutive memory words,
from main memory. Accessing nearby memory locations is therefore much faster than accessing scattered locations.

Accessing matrix elements in row-major order (left-to-right, top-to-bottom) is cache-friendly because consecutive elements in a row are already loaded in the cache.
Conversely, accessing matrix elements column-by-column is cache-unfriendly because each access may involve a different cache line.

**Implications for Matrix Multiplication**

In the standard matrix multiplication algorithm ``C = A × B``, where ``C[i][j] = Σₖ A[i][k] × B[k][j]``:
Assuming i, j, k loop order:
- Accessing matrix **A** is cache-friendly because consecutive elements in memory are accessed one after another for two consecutive values of inner loop iterator ``k``.
- Accessing matrix **B** is not cache-friendly because memory accesses skip whole matrix rows for two consecutive values of ``k``.

This asymmetry significantly impacts performance and motivates various optimization techniques such as loop reordering or tiling.

Linear Transformation and Computational Flexibility
===================================================

Matrix multiplication represents a linear transformation where the computation can be viewed as a sequence of dot products.
This linearity allows significant computational flexibility, such as:

1. Loop Reordering: Changing the order of loops does not impact the result, but can impact performance on architectures with cache hierarchies.
2. Loop Tiling: A subset of the resulting matrix can be calculated at a time, improving cache locality and performance.
3. Parallelization: Different matrix regions can be computed concurrently

Note that operation ordering can impact result when floating point operations are not associative, which is the case for most modern architectures.
In this lab, we will ignore this issue, but you should be aware of it, especially in cases when the values in matrices are of significantly different orders of magnitude.


Loop Tiling
-----------

Loop tiling (a.k.a. loop blocking) is a loop transformation technique that divides the loop into smaller chunks.
On architectures with cache hierarchies, this can lead to improved performance by reducing cache misses, especially when combined with other optimizations like loop reordering.
It is worth noting that loop tiling by itself may lead to reduced performance if the tile size is not chosen optimally or if tiling causes the compiler to generate code that is not optimal for the target architecture.
On the other hand, some architectures offer hardware support for vector or matrix operations of limited vector/matrix sizes.
When targeting such architectures, loop tiling can be used to divide work into smaller chunks that map to the underlying hardware.
As we will see later, the Tenstorrent Tensix processor has hardware support for tiled operations, which is our main motivation for exploring tiling.

Simple Tiling Example
^^^^^^^^^^^^^^^^^^^^^

Original Doubly Nested Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Original loop without tiling
   for (int i = 0; i < M; i++)
       for (int j = 0; j < N; j++)
           some_computation(i, j);

Tiled Version
~~~~~~~~~~~~~

.. code-block:: cpp

   // Tiled version with M-N blocking
   int M_TILE_SIZE = 32;
   int N_TILE_SIZE = 32;

   // Assumes M and N are divisible by tile sizes
   int num_row_tiles = M / M_TILE_SIZE;
   int num_col_tiles = N / N_TILE_SIZE;

   for (int row_tile = 0; row_tile < num_row_tiles; row_tile++)
       for (int col_tile = 0; col_tile < num_col_tiles; col_tile++)
           // Inner tile computation
           for (int row = row_tile * M_TILE_SIZE; row < (row_tile + 1) * M_TILE_SIZE; row++)
               for (int col = col_tile * N_TILE_SIZE; col < (col_tile + 1) * N_TILE_SIZE; col++)
                   some_computation(row, col);


Tiled Matrix Multiplication
===========================

In this part of the lab, you will implement two versions of matrix multiplication: a straightforward triply-nested loop, and a tiled version.
The triply-nested loop version is simply a reference implementation provided at the beginning of the lab.
The tiled version should be implemented as follows:

1. Input to the matrix multiplication should be a vector to ensure data is contiguous in memory.
2. Create a function ``tile_matmul`` that multiplies a single tile
3. Implement a main matrix multiplication function using tiling and then calling ``tile_matmul`` for each tile.
4. Allow parameterization of tile size
5. Verify the correctness of the implementation by comparing the result with the reference implementation.
6. Profile the performance of the implementation and compare it with the reference implementation.
   Make sure to compile with -O3 optimization level when comparing performance.



.. code-block:: cpp

    // Single tile matrix multiplication
    void tile_matmul(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        int K,
        int N,
        int row_offset, // Tile starting indices
        int col_offset,
        int k_offset,
        int TH, // Tile height
        int TW // Tile width
    ) {
        // Implement multiplication of TH x TW tile by TW x TH matrix to produce TH x TH result.
        // Use the starting indices to index into matrices A, B and C.
        // Tile in A begins at (row_offset, k_offset) and has size TH x TW.
        // Tile in B begins at (k_offset, col_offset) and has size TW x TH.
        // Resulting tile in C begins at (row_offset, col_offset) and has size TH x TH.
        // Accumulate the result into C (assume it is initialized to 0).
        // Hint: your code will be a lot simpler if you create a helper function get_idx to index
        // into a matrix using row and column coordinates and number of columns.
    }

    std::vector<float> tiled_matrix_multiply(
        const std::vector<float>& A,
        const std::vector<float>& B,
        int M, // Full matrix dimensions
        int K,
        int N,
        int TH, // Tile height
        int TW // Tile width
    ) {
        // Implement full matrix multiplication using tiling
    }





Introduction to Tenstorrent Architecture
========================================

Tenstorrent's devices are a line of AI accelerator devices that are typically delivered as PCIe cards attached to a standard host.
In this model, the host CPU runs a standard C++ host program. In the host program, developers can use the TT-Metalium C++ API to configure the accelerator,
allocate memory on the device, and dispatch kernels, which are also written in C++, to Tensix cores.
PCIe card contains one or more Tensix devices, each device consisting of a Tensix processor and a dedicated DRAM.
Note that the device DRAM is separate from the system (host) DRAM and explicit communication is required to transfer data between them.
The Tensix processor contains an array of Tensix cores with network on-chip (NoC) to pass data from DRAM to fast on-chip SRAM memory and between cores.
On-Chip SRAM memory is often referred to as L1, but that is a misnomer because this memory does not operate like a cache memory.
Each Tensix core has a dedicated compute unit specifically optimized for matrix and vector operations on tensors.

The host program first transfers tensors from host memory into buffers in the device DRAM.
Tensix cores then move data from DRAM into on-chip SRAM, perform matrix and vector operations using dedicated hardware units and may store intermediate
results in on-chip SRAM or device DRAM.
Once all the required computation is done, the results are moved back to host memory for further processing, verification, or I/O.

A key design principle of Tenstorrent architecture is that most computation steps occur entirely on the device, with explicit and carefully orchestrated
data movement between DRAM and Tensix SRAM, and between neighboring cores over the NoC.
Typical workloads move input data from the host to the accelerator once, perform many layers or time steps directly on Tensix cores while reusing
local SRAM-resident data, and only transfer compact outputs or checkpoints back to the host.
When used this way, host-device communication over PCIe represents a relatively small fraction of the total runtime,
allowing the accelerator's on-card compute and data-movement engines to dominate performance.

In this lab we will focus on programming a single Tensix processor in TT-Metalium and descriptions of the architectural features will be limited to this context.
We will extend the architectural descriptions and programming model to multiple Tensix processors in subsequent labs.

Tile-Based Architecture
-----------------------

The Tenstorrent architecture is a tile-based architecture.
This means that the data is processed in tiles, which are commonly 32x32 blocks of data.
Similar to vector architectures, which can perform an operation on one or more vectors of data using a single instruction,
a Tensix core can perform matrix operations on one or more tiles in one instruction.
For example, a Tensix core can perform a matrix multiplication on two tiles each having 32x32 elements and produce a 32x32 result tile in one instruction.

Memory Layout and Tiling
========================

Most applications deal with data that is larger than 32x32 tiles, so we need to find a way to serve the data to Tensix cores in tile-sized chunks.
In the previous exercise you performed loop tiling, which changed memory access pattern without changing the underlying data layout,
which was still stored in row-major order.
An alternative approach is to change the data layout itself by placing all elements of a tile in memory contiguously.
This **tiled memory layout** is the main memory layout used by the Tenstorrent architecture.

Consider the following 9x4 matrix:

.. code-block:: cpp

   [a00  a01  a02  a03]
   [a10  a11  a12  a13]
   [a20  a21  a22  a23]
   [a30  a31  a32  a33]
   [a40  a41  a42  a43]
   [a50  a51  a52  a53]
   [a60  a61  a62  a63]
   [a70  a71  a72  a73]
   [a80  a81  a82  a83]

In row-major layout, this matrix is stored in memory as:

.. code-block:: cpp

   [a00][a01][a02][a03][a10][a11][a12][a13][a20] ... [a80][a81][a82][a83]

In tiled memory layout with tile size 3x2, this matrix is stored in memory as:

.. code-block:: cpp

   [a00][a01][a10][a11][a20][a21][a02][a03][a12][a13][a22][a23][a30] ... [a82][a83]

.. note::
   Add an image of a tiled memory layout here.

Observe that all elements of a tile are stored contiguously in memory. In this tiled layout there are two "second-order" row-major orderings:

1. Elements within each tile are stored in row-major order.
2. Ordering of tiles relative to other tiles follows row-major ordering. That is, the first tile is stored at the beginning of the memory
   followed by the tile to its right, and so on until the last tile in the row of tiles. Then the next row of tiles is stored,
   starting with the tile in the first column of the next row of tiles.


Tensix Programming Model
========================

Tenstorrent devices can be programmed at multiple abstraction levels, from high-level neural network libraries to low-level kernel development.
At the highest level, TT-NN provides a PyTorch-like Python API for neural network operations, while TT-Metalium serves as the low-level programming
model that gives developers direct control over the Tensix hardware architecture.

TT-Metalium is unique because it exposes the distinctive architecture of Tenstorrent's Tensix processors.
Unlike traditional GPUs that rely on massive thread parallelism, TT-Metalium enables a cooperative programming model where developers typically
write three types of kernels per Tensix core: reader kernels for data input, compute kernels for calculations, and writer kernels for data output.
These kernels coordinate through circular buffers in local SRAM, enabling efficient pipelined execution that overlaps data movement with computation.

TT-Metalium's compute API abstraction layer maintains kernel code compatibility across different hardware generations while ensuring optimal performance.
This means the same kernel code can run efficiently on different Tenstorrent processor generations without requiring developers to rewrite code
for each architecture.
TT-Metalium serves as the foundation for higher-level frameworks, providing the base layer for all Tenstorrent software development.

Before we look at a Tensix program, it's important to emphasize a fundamental distinction is between the **host** and the **device**.
The **host** is the conventional CPU system (often x86 or ARM) where the main application runs.
The **device** is the accelerator: a mesh of Tensix cores with their own on-chip memories and access to off-chip DRAM
(recall that this is **device DRAM**, distinct from **host DRAM**).
Tensix cores execute relatively small, specialized programs called kernels that are designed for high-throughput numeric computation,
not for general-purpose tasks.

Crucially, the **host and device live in different address spaces and have different execution models**.
A pointer or object on the host (e.g., an array in system DRAM or a C++ object) is not directly visible to a Tensix core.
To use data on the device, the host must explicitly allocate accelerator-side memory (e.g., DRAM pages or on-chip buffers),
transfer data to that memory, and then pass information about data layout and addresses of memory allocated in
device memory address space down to the kernels.
Conversely, when a kernel finishes, any results you want on the host must be explicitly copied back.

There is also a **two-stage view of "compile time" versus "run time" that spans host and device**.
Your host program is compiled ahead of time by a standard compiler, while Tensix kernels are compiled just-in-time (JIT) by the
runtime while the host program is running. At that kernel-compile stage, certain parameters (like tensor layout, tile sizes) are
known and are considered compile-time constants as far as kernel code is concerned, and hence they end up in the kernel binary.
The advantage of this is that kernel code can be aggressively optimized.
At execution time, other parameters (such as specific base addresses in device DRAM or the number
of tiles to process in a particular launch) are provided as runtime arguments.
Based on this description, it may seem that we should specify as many parameters as possible at compile time.
However, this is not always possible or practical. In some cases we may choose to pass some information as runtime arguments
even if it would be possible to determine this information at compile time. In some cases, letting the runtime determine this information
may provide more flexibility. In other cases, we may want to reuse the same kernel code for different parameter values, rather than
producing a new kernel binary for each combination of parameter values. In such cases, the choice of which parameters to pass at compile time
and which to pass at runtime is a trade-off between performance and flexibility.

Understanding which information lives on the host, which is baked into the kernel binary, and which is supplied dynamically at
launch is key to reasoning about performance, correctness, and why the APIs are structured the way they are.


Example Tensix Program
======================

We will now present a simple example Tensix program that performs an elementwise addition of two tensors of shape MxN.
This program will be used to illustrate the Tensix programming model, different types of kernels, and how they map to the underlying architecture.
Key points will be highlighted in this text. Detailed comments are provided in the C++ code in this and other files to help with code understanding.

The main program for the code example being discussed is located in the file ``tt_metal/programming_examples/lab_eltwise_binary/lab_eltwise_binary.cpp``.
First thing to emphasize is that all the code in this file executes on the host, although there are many API calls that cause activity on the device.

Looking at the main function, we see that the host program first initializes input data for the operation and performs a reference computation on the host CPU.
This will be used to verify the correctness of the Tensix implementation. Note that the data type used is bfloat16 (brain floating point), which is a
16-bit floating point format commonly used in AI applications. Since the host CPU doesn't natively support bfloat16,
we use the `bfloat16` class from the `tt-metalium` library and cast data between this type and single-precision (32-bit) floating point as needed.

Next, the host program initializes the Tensix device and program state by calling the function `init_program`.
This function contains a lot of boilerplate code that configures the device to run our code on a single Tensix core. Most programs utilizing
a Tensix device would use similar code to configure the device and program, and exact initialization details are not important for this lab.

After initialization, the program calls the function `eltwise_add_tensix`, which is the main function that configures and creates kernels
and triggers elementwise addition on the Tensix device.
Finally, the program validates the results by comparing the Tensix output with the reference computation on the host CPU.


Kernel Types and Data Flow
--------------------------

Programming with Metalium typically requires three kernel types per Tensix core: a **reader kernel** for data input,
a **compute kernel** for calculations, and a **writer kernel** for data output. These kernels coordinate through circular buffers in SRAM.
The circular buffers act as producer-consumer queues, enabling safe and efficient data exchange between kernels.
Note that the circular buffers typically contain only a small number of tiles at a time, not the entire tensor.
Also note that reader kernels and writer kernels are commonly referred to as data movement kernels.

.. image:: images/tenstorrent-circular-buffer-send-data-cross-kernel-or-itself.webp
   :width: 900
   :alt: Circular buffer data flow

   Figure 1: Kernel data flow through circular buffers

!!!!
NOTE: THIS IMAGE HAS PROBLEMS!!!
Writer should be Kernel 1 not Kernel 0.
"Think them as pipes!!!"
!!!!

Each kernel interacts with the buffers as follows:

- **Reader kernel:** Reads data (e.g. from device DRAM) into the circular buffer and signals when new data has been read and is available.
- **Compute kernel:** Waits for data to become available in the buffer before processing it. After computation, it writes the results to another
circular buffer and marks data as ready in that buffer.
- **Writer kernel:** Waits for the computed results to appear in the buffer before writing them to the output location (e.g. device DRAM).

This mechanism ensures that each kernel only proceeds when the necessary data is ready, preventing race conditions and enabling asynchronous,
pipelined execution across the hardware. Different kernel types are mapped to the Tensix core, whose high-level diagram is shown in Figure 2.

.. figure:: images/tensix_core.png
   :width: 600
   :alt: Top-level diagram of Tensix Core

   Figure 2: Top-level diagram of Tensix Core

Tensix Core consists of four major parts:

1. Internal SRAM (L1) Memory - Stores input/output tiles in circular buffers for fast access by the Tensix engine.
   It also holds program code for all RISC-V processors within the core.
2. Two Routers - Manage data movement between device DRAM and internal SRAM (L1) memory.
3. Tensix Engine - Hardware accelerator that efficiently performs matrix and vector computations on tiles.
4. Five RISC-V Processors that control the Tensix Engine and routers:
   - RISC-V 0 and RISC-V 4 - These processors control routers to exchange data between the Internal SRAM and device DRAM (or other Tensix cores).
   Either of these can be used for reader or writer kernel.
   - RISC-V 1 through RISC-V 3 - These processors control the Tensix Engine through specialized Tensix instructions.
   Note that these RISC-V processors don't perform actual tile computations.
   Instead, they serve as microcontrollers directing the operations of the Tensix Engine. One RISC-V processor is responsible for issuing commands to
   the compute engine, while the other two are responsible for transferring tile data between circular buffers in SRAM and Tensix Engine registers.
   Compute kernel code defines functionality for all three of these processors.


Kernel Creation and Configuration
---------------------------------

The function `eltwise_add_tensix` creates and configures the kernels that will be used to perform the elementwise addition on the Tensix device.
To understand this code, we need to understand the different types of kernels and how they map to the Tenstorrent architecture.


The first step is to create a program object. A program is a collection of kernels that are executed on the device.
Kernels will be specified later.

Next, the program creates a circular buffer for the input and output data.
Circular buffers are a type of memory that is used to store data that is being transferred between the host and the device.
They are a key feature of the Tensix programming model and are used to manage data movement between the host and the device.







Scrap Heap
==========

(LEFTOVER TEXT THAT MAY COME HANDY LATER)


Terminology
-----------

Device
------
Each device contains a number of Tensix cores and a dedicated DRAM, which is separate from the system (host) DRAM.
Device in Tensix documentation can often refer to both the Tensix processor and its accompanying DRAM.

Tensix Core
-----------

The Tensix core is the main processing unit of the Tensix processor.
It is a custom-designed processor for matrix multiplication and other linear algebra operations.

Mention that kernels are JIT compiled and errors will not be caught at compile time (and also no rebiuilding required if only kernel code is changed).

Show example TT program that does something simple (e.g. elementwise operation) to illustrate the flow.

Walk through binary OP Example
(Use Cursor to transform existing example into  "tensor" version)

Emphasize how async read only needs the number of the tile to read, not the actual address.


Intentionally introduce a bug or two and then ask students to use TT-specific debugging features to debug and find the issue.
Main purpose here is for students to learn to use these debugging features

Ask students to transform the example program to single core matrix multiply

Profile its performance so it can be used for comparison against future labs.


Memory Layout and Tiling
========================

Loop tiling changes memory access pattern without changing the underlying data layout.
An alternative approach is to change the data layout itself by placing all elements of a tile in memory contiguously.
This is called **tiled memory layout** and is the main memory layout used by the Tenstorrent architecture.

Consider the following matrix:

.. code-block:: cpp

   [a00  a01  a02  a03]
   [a10  a11  a12  a13]
   [a20  a21  a22  a23]
   [a30  a31  a32  a33]
   [a40  a41  a42  a43]
   [a50  a51  a52  a53]
   [a60  a61  a62  a63]
   [a70  a71  a72  a73]
   [a80  a81  a82  a83]

In row-major layout, this matrix is stored in memory as:

.. code-block:: cpp

   [a00][a01][a02][a03][a10][a11][a12][a13][a20] ... [a80][a81][a82][a83]

In tiled memory layout with tile size 3x2, this matrix is stored in memory as:

.. code-block:: cpp

   [a00][a01][a10][a11][a20][a21][a02][a03][a12][a13][a22][a23][a30] ... [a82][a83]

.. note::
   Add an image of a tiled memory layout here.
