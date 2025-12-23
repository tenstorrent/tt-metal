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


Exercies 1: Tiled Matrix Multiplication
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

Most applications deal with data that is larger than a single 32x32 tile, so we need to find a way to serve the data to Tensix cores in tile-sized chunks.
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

Note that the number of rows/columns in a matrix may not always evenly divide the number of rows/columns in a tile.
In this case, the matrix needs to be padded with data to the next tile boundary.
This data must not impact the result of the computation, so depending on the operation, it may be padded with zeros or some other value.
In this lab, we will assume that all matrices are aligned to tile boundaries.

With this memory layout, code can now read one tile at a time and find the next tile in memory at a fixed offset rather than having to
assemble a tile by merging different sections of memory, as would be the case for row-major memory layout.
This allows for efficient memory access and computation.


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
The host program is compiled ahead of time by a standard compiler, while Tensix kernels are compiled just-in-time (JIT) by the
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

Before diving into the function `eltwise_add_tensix`, let us discuss the different types of kernels and how they map to the underlying hardware.
Programming with Metalium typically requires three kernel types per Tensix core: a **reader kernel** for data input,
a **compute kernel** for calculations, and a **writer kernel** for data output.
Reader and writer kernels are collectively referred to as data movement kernels.
Data movement and compute kernels communicate through circular buffers in internal SRAM.
The circular buffers act as producer-consumer FIFO (First In First Out) queues, enabling safe and efficient data exchange between kernels.
Note that the circular buffers typically contain only a small number of tiles at a time, not the entire tensor.

.. figure:: images/tenstorrent-circular-buffer-send-data-cross-kernel-or-itself.webp
   :width: 900
   :alt: Kernel data flow through circular buffers
   :align: center

   Figure 1: Kernel data flow through circular buffers

.. note::
   **Note:** This image has known issues. Writer should be Kernel 1, not Kernel 0. Think of them as pipes!

Each kernel interacts with the buffers as follows:

- **Reader kernel:** Reads data (e.g. from device DRAM) into the circular buffer and signals when new data has been read and is available.
- **Compute kernel:** Waits for data to become available in the buffer before processing it. After computation, it writes the results to
  another circular buffer and marks data as ready in that buffer.
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
At a high-level, the function creates a tensor object that resides in device DRAM and then creates two dataflow kernels, one reader and one writer,
one compute kernel, and three circular buffers to pass data between the kernels, and then triggers kernel execution on the device.

The function creates three Tensor objects of shape MxN using the tile layout described earlier.
These tensors are allocated in device DRAM, which is distinct from host DRAM and is directly attached to the Tensix processor.
The input tensors are created and initialized by transferring the data from the host to the device in one step using `Tensor::from_vector`.
The vectors passed to `Tensor::from_vector` are the same input vectors that were used for the reference computation.

The function then creates three circular buffers to enable data movement between kernels.
A circular buffer is a FIFO buffer with configurable size.
Creating a circular buffer simply means allocating sufficient internal SRAM memory based on specified configuration, and associating
specified circular buffer index with this SRAM memory and configuration.
In our example program, circular buffers are created with two tiles each to allow for double buffering. For example, reader kernel can be reading one tile
while the compute kernel is processing the other tile, enabling pipelined execution.
Number of tiles in a circular buffer can be adjusted to trade off memory for performance, but generally there are diminishing
returns observed after several tiles.

The function creates the three types of kernels discussed earlier: reader, compute, and writer.
Each kerrnel can take two types of arguments: compile-time and runtime kernel arguments, as mentioned earlier.

.. _kernel-args-blurb:

The two types of kernel arguments differ in *when* their values are determined and *how* they are used.

**Compile-time kernel arguments**

* Values that are known when the kernel is built (JIT-compiled).

* They are hard-coded into the kernel binary and can be used by the compiler to specialize and optimize the code, by e.g. unrolling loops, removing branches, choosing specific data paths, etc.

* Changing a compile-time argument effectively means generating a new version of the kernel binary.

* These arguments are provided by the host during kernel configuration as ``compile_args``

**Runtime kernel arguments**

* Values that are provided by the host each time the kernel is launched, possibly varying per core.

* Stored in a small argument block in the core's local memory and read by the kernel at the start of execution, for example using ``get_arg_val<T>(index)``.

* They do not change the compiled binary; instead, they affect kernel behavior at run time. Examples include buffer base addresses,
  flags to enable/disable certain features, etc.

* Same compiled kernel can be reused many times with different runtime arguments, without recompiling.

In summary, compile-time arguments specialize the kernel *code itself*, while runtime arguments specialize *what that code does on a particular launch*.

In our example program, reader and writer kernels take information about tensor layout and data distribution as compile-time arguments.
Compile-time arguments are passed as a vector of uint32_t values. TensorAccessorArgs utility is a clean way to append relevant tensor layout
information into this vector, without programmer having to worry about internal details.

Dataflow kernels take the base addresses of the input and output buffers in device DRAM, along with the number of tiles to process
as runtime arguments.

Compute kernel takes the number of tiles to process as a compile-time argument and doesn't take any runtime arguments.
At first, it may seem like an odd choice to pass the number of tiles as a compile-time argument to the compute kernel,
but as a runtime argument to dataflow kernels.
Since using compile-time arguments enables various compiler optimizations, it is particularly suitable to use them for compute kernels, which are compute bound.
While passing the number of tiles as a compile-time argument to dataflow kernels would also work, they may not benefit much since their performance is memory bound.
At the same time, using compile-time arguments for dataflow kernels would cause them to be recompiled for each different number of tiles.
since the performance benefit of using compile-time arguments is not significant for dataflow kernels,
we optimize for code reuse and avoid recompilation by using runtime arguments for dataflow kernels.
This distinction will become more apparent in subsequent labs when we start working with multiple Tensix cores.

Creating kernels registers the kernels with the program object, so that they can be executed later.

Finally, the function executes the kernels by adding the program to the workload and enqueuing it for execution, which triggers kernel JIT compilation
followed by kernel execution on the device. It is useful to remind ourselves that unitl this point, all the code we discussed executed on the host,
not on the device.
We will examine kernel code next.

Reader Kernel Code
------------------

The function can be summarized by the following pseudo-code:
```cpp
read_runtime_arguments()
read_compile_time_arguments()
create_address_generators()
for (i in range (0 .. n_tiles) {
    transfer_tile_from_dram_to_circular_buffer(in0, i)
    transfer_tile_from_dram_to_circular_buffer(in1, i)
}
```

The reader kernel in ``tt_metal/programming_examples/lab_eltwise_binary/kernels/dataflow/read_tiles.cpp`` is responsible for transferring data
from device DRAM into circular buffers located in internal device SRAM, where it can be efficiently accessed by the compute kernel.
The kernel reads the base addresses of the two input tensors in DRAM and the total number of tiles to process as runtime arguments.

The kernel uses two circular buffers (``c_0`` and ``c_1``) as destination buffers for the two input tensors.
It retrieves the tile size from the circular buffer configuration, which must match the tile size used in the DRAM buffers.

``TensorAccessorArgs`` objects whose information was placed into compile-time arguments vector by the host program, is
now reassembled into device-side ``TensorAccessorArgs`` objects that describe the tensor's layout in memory
These objects are then combined with the runtime base addresses to create address generator objects (``TensorAccessor``).
These address generators abstract away the complexity of physical memory layout, such as data distribution among DRAM banks
by automatically computing the physical DRAM address for any given tile index.

The main processing loop iterates over all tiles, implementing a producer-consumer pattern with the compute kernel.
For each tile, the kernel first reserves space in both circular buffers using blocking calls to ``cb_reserve_back``
to ensure that space is available before attempting to write.
Once space is reserved, the kernel obtains write pointers to the circular buffers and initiates two non-blocking asynchronous
read operations using ``noc_async_read_tile``. Observe that this call takes in the index of the tile being read and address generator,
along with the circular buffer address to write the tile to. Address generator automatically maps the logical tile index to the correct physical
DRAM address based on the specific memory layout.

Because ``noc_async_read_tile`` is non-blocking, the two reads can proceed in parallel if sufficient bandwidth is available, transferring data from DRAM to
the circular buffers simultaneously. After both reads are initiated, the kernel calls ``noc_async_read_barrier`` to wait
for both transfers to complete. This is important because the kernel should not signal that the tiles are ready for consumption
until data is actually available.

The reader kernel repeats this process for all tiles. Given that circular buffers were created with two tiles each, the reader kernel can
read a new tile while the compute kernel is processing the previous one.

Compute Kernel Code
-------------------

The compute kernel in ``tt_metal/programming_examples/lab_eltwise_binary/kernels/compute/tiles_add.cpp`` is responsible for performing the elementwise addition of two tiles.

The function can be summarized by the following pseudo-code:
```cpp
read_compile_time_arguments()
initialize_tensix_engine_for_elementwise_addition()
for (i in range (0 .. n_tiles) {
    add_tiles_in_input_circular_buffers()
    write_result_to_output_circular_buffer()
}
```

The kernel reads the number of tiles to process as a compile-time argument, enabling compiler optimizations such as loop unrolling.

An important architectural detail is that the compute kernel actually runs on three different RISC-V processors within the Tensix core:
unpacker, compute, and a packer.
The compiler automatically generates appropriate code for each of these three processors from the same source code, relieving the programmer
from having to write different code for each processor.
The unpacker handles reading data from circular buffers, the compute processor performs the actual arithmetic operations using the
Floating Point Unit (FPU) of the Tensix engine, and the packer writes results back to circular buffers.
It is worth repeating that these RISC-V processors don't perform actual computations or packing/unpacking operations.
They simply issue commands to the Tensix engine to perform the actual computations and packing/unpacking operations.

The compute kernel uses circular buffers ``c_0`` and ``c_1`` for the two input tensors and ``c_16`` for the output tensor.
There are 32 circular buffers in total (0-31), and the exact indices used are up to programmer's discretion, provided they are used consistently
(i.e. reader and writer kernels must use the corresponding indices as the compute kernel).

The kernel initializes the FPU for elementwise binary operations, first calling ``binary_op_init_common``
to set up the general binary operation infrastructure, followed by ``add_tiles_init`` to configure the FPU specifically for addition.
This initialization only needs to be done once before the main loop, since all tiles use the same operation.

The main processing loop iterates over all tiles.
For each tile, the kernel first waits for one tile to become available in each input circular buffer using blocking calls to ``cb_wait_front``.
These blocking calls ensure that the reader kernel has finished transferring the data before the compute kernel attempts to use it.
The kernel then acquires access to the destination register array using ``tile_regs_acquire`` and calls ``add_tiles`` to perform the elementwise addition.
The destination register array is a special storage area in the FPU that can hold multiple tiles and serves as the temporary
output location for FPU computations. The acquire operation also initializes all tiles in the destination register array to zero,
which is not important for our example, but is useful for operations like matrix multiplication where results accumulate.

After the computation completes, the kernel marks the input tiles as consumed using ``cb_pop_front`` to free space in the circular buffers,
then releases the destination register to signal that the compute core has finished writing, which allows packer to read the result.

The packer core for its part waits for the destination register to be ready using ``tile_regs_wait``,
ensures there is space in the output circular buffer, and then copies the result from the destination register to the output circular buffer using ``pack_tile``.
Finally, it marks the output tile as ready using ``cb_push_back`` and releases the destination register using ``tile_regs_release``.

While it may seem like some of these operations are redundant (e.g. waiting on the destination register when it has seemingly just been released),
it is important to remember that compute kernel code is executed on three different RISC-V processors.
This synchronization mechanism using acquire, commit, wait, and release ensures that the three RISC processors coordinate properly,
with the compute processor writing results and the packer processor reading them without conflicts.


Writer Kernel Code
------------------

The function can be summarized by the following pseudo-code:
```cpp
write_tiles(c_16, n_tiles);
```

The writer kernel in ``tt_metal/programming_examples/lab_eltwise_binary/kernels/dataflow/write_tiles.cpp`` is responsible for transferring computed results from circular buffers in internal device SRAM back to device DRAM, where they can eventually be read back to the host.
The kernel reads two runtime arguments: the base address of the output tensor in DRAM and the total number of tiles to write.
These runtime arguments enable the same compiled kernel to work with different output tensor locations without recompilation.

The kernel uses circular buffer ``c_16`` as the source buffer containing the computed results produced by the compute kernel.
It retrieves the tile size from the circular buffer configuration, which must match the tile size used in the DRAM buffers.
Similar to the reader kernel, the writer kernel creates an address generator object using ``TensorAccessor`` to determine where each tile should be written in DRAM.
The address generator is constructed by extracting tensor layout parameters from compile-time arguments using ``TensorAccessorArgs``, then combining them with the runtime base address.
This allows the address generator to automatically compute the correct physical DRAM address for any given tile index, abstracting away the complexity of the tiled memory layout and data distribution.

The main processing loop iterates over all tiles, implementing a consumer pattern that complements the compute kernel's producer role.
For each tile, the kernel first waits for a tile to become available in the circular buffer using the blocking call ``cb_wait_front``.
This ensures that the compute kernel has finished producing the tile before the writer attempts to read it.
Once a tile is available, the kernel obtains a read pointer to the circular buffer and initiates a non-blocking asynchronous write operation using ``noc_async_write_tile``.
This call takes the tile index, the address generator (which maps the logical index to the physical DRAM address), and the circular buffer read address.
After initiating the write, the kernel calls ``noc_async_write_barrier``, which is a blocking call that waits until the write operation completes.
This synchronization is critical to ensure that data has actually been written to DRAM before the kernel proceeds.
Finally, the kernel marks the tile in the circular buffer as consumed by calling ``cb_pop_front``, which frees up space in the circular buffer for the compute kernel to produce the next tile.
This coordination between the compute and writer kernels enables pipelined execution, where computation and data movement can overlap.




TODO: Have a diagram which shows what resides where (device, host, CBs, DRAM,...)

Kernel Compilation and Execution
--------------------------------

As mentioned earlier, kernels are JIT compiled and executed on the device. This presents both advantages and disadvantages during development.
On the one hand, if one updates only kernel code, there is no need to rebuild before running the program to test that the changes had desired effect.
On the other hand, it also means that errors in the kernel code will not be caught at host-code compile time, but only at time of host code execution,
when JIT compilation is triggerred.


Exercise 2: Debugging Tensix Kernels
====================================

Introduce the following bugs:

Introduce a syntax error in the kernel code and rerun the program to see how JIT compilation errors are reported.


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
