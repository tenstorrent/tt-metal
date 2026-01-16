Lab 1: Single Core Matrix Multiplication
########################################

Introduction
************

Matrix multiplication is a fundamental linear-algebra operation used in scientific computing, machine learning, and graphics.
The standard algorithm multiplies two input matrices to produce a third matrix.

Basic Algorithm
===============

Given two matrices ``A`` of shape ``MxK`` and ``B`` of shape ``KxN``, the resulting matrix ``C`` will have shape ``MxN``.
The classical matrix multiplication algorithm can be expressed using a triple-nested loop structure,
where each element of the resulting matrix is computed as the dot product of the corresponding row of ``A`` and the corresponding column of ``B``:

.. code-block:: cpp

   for (int i = 0; i < M; i++) {
       for (int j = 0; j < N; j++) {
           C[i][j] = 0.0;
           for (int k = 0; k < K; k++) {
               C[i][j] += A[i][k] * B[k][j];
           }
       }
   }

The naive implementation has a time complexity of O(M*N*K).


Row-Major Memory Layout
***********************

Memory space is a linear (one-dimensional) array of memory words.
However, we often need to represent data with more dimensions, such as two-dimensional matrices or n-dimensional tensors.
In C++, two-dimensional arrays are usually stored in memory using **row-major order**.
This means that elements of each row are stored contiguously in memory, with rows placed one after another.

Consider a ``3x4`` matrix ``A``:

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/a_matrix_3x4.jpg
   :width: 250
   :alt: A 3x4 Matrix A
   :align: center


Using row-major layout, this matrix is stored in memory as:

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/matrix_3x4_row_major.jpg
   :width: 750
   :alt: A 3x4 Matrix A in Row-Major Layout
   :align: center

When accessing element ``A[i][j]`` in a matrix with ``N`` columns, the memory address is calculated as::

   address = base_address + (i * N + j) * sizeof(element)

Cache-Friendly Access Patterns
==============================

Modern CPUs fetch data in cache lines of consecutive words, so accessing nearby addresses is much faster
than scattered accesses. Accessing matrix elements in row-major order (left-to-right, top-to-bottom) is
cache-friendly because consecutive elements in a row are already loaded in the cache.
Conversely, accessing matrix elements column-by-column is inefficient because each access may
trigger a cache miss by requesting a different cache line.

Implications for Matrix Multiplication
--------------------------------------

In the standard matrix multiplication algorithm ``C = A * B``, where ``C[i][j] = ∑ₖ A[i][k] * B[k][j]``,
and assuming ``i``, ``j``, ``k`` loop order:

* Accessing matrix ``A`` is cache-friendly because consecutive elements in memory are accessed
  one after another for two consecutive values of the inner loop iterator ``k``.
* Accessing matrix ``B`` may degrade performance because memory accesses skip an entire matrix row
  for two consecutive values of ``k``.

This asymmetry significantly impacts performance and motivates various optimization techniques such as loop reordering or tiling.

Linear Transformation and Computational Flexibility
***************************************************

Matrix multiplication represents a linear transformation where the computation can be viewed as a sequence of dot products.
This linearity allows significant computational flexibility, such as:

1. Loop Reordering: Changing the order of loops does not affect the result, but can impact performance on architectures with cache hierarchies.
2. Loop Tiling: A subset of the resulting matrix can be calculated at a time, improving cache locality and performance.
3. Parallelization: Different matrix regions can be computed simultaneously.

Note that changing the operation order can affect the result because floating-point arithmetic is not associative.
In this lab, we ignore this issue, but be aware of it when matrix values differ greatly in magnitude.

Loop Tiling
***********

Loop tiling (a.k.a. loop blocking) is a loop transformation technique that splits loops into smaller chunks.
On architectures with caches, this often improves performance by reducing cache misses, especially when combined with loop reordering.
Poor tile sizes, however, can make tiled loops slower instead of faster.

Some architectures offer hardware support for vector or matrix operations of limited vector/matrix sizes.
When targeting such architectures, loop tiling can be used to divide work into smaller chunks that map to the underlying hardware.
As we will see later, the Tenstorrent Tensix processor has hardware support for tiled operations, so understanding tiling is important
when programming Tensix cores.

Simple Tiling Example
=====================

Original Doubly Nested Loop
---------------------------

.. code-block:: cpp

   // Original loop without tiling
   for (int i = 0; i < M; i++) {
       for (int j = 0; j < N; j++) {
           some_computation(i, j);
       }
   }

Tiled Version
-------------

.. code-block:: cpp

   // Tiled version with MxN tiling
   constexpr int M_TILE_SIZE = 32;
   constexpr int N_TILE_SIZE = 32;

   // Assumes M and N are divisible by tile sizes
   const int num_row_tiles = M / M_TILE_SIZE;
   const int num_col_tiles = N / N_TILE_SIZE;

   for (int row_tile = 0; row_tile < num_row_tiles; row_tile++) {
       for (int col_tile = 0; col_tile < num_col_tiles; col_tile++) {
           // Inner tile computation
           for (int row = row_tile * M_TILE_SIZE; row < (row_tile + 1) * M_TILE_SIZE; row++) {
               for (int col = col_tile * N_TILE_SIZE; col < (col_tile + 1) * N_TILE_SIZE; col++) {
                   some_computation(row, col);
               }
           }
       }
   }


Exercise 1: Tiled Matrix Multiplication
=======================================

In this part of the lab, you will implement two versions of matrix multiplication:
a straightforward triply-nested loop, and a tiled version.
The triply-nested loop version is simply a reference implementation provided at the beginning of the lab.
For this exercise, write standard C++ code that can be compiled and run on any general-purpose CPU.

The tiled version should be implemented as follows:

#. Input to the matrix multiplication should be a one-dimensional vector to ensure data is contiguous in memory.
#. Create a function ``tile_matmul`` that multiplies a single tile.
   Write your code so that the function can be called with different tile sizes.
#. Implement a main matrix multiplication function using tiling and then calling ``tile_matmul`` for each tile.
#. Use the tiled implementation to multiply matrix ``A`` of size ``640x320`` with
   matrix ``B`` of size ``320x640`` to produce matrix ``C`` of size ``640x640``. Use tile size ``32x32``.
#. Verify correctness of the tiled implementation by comparing the result with the reference implementation.
#. Measure wall-clock time of the tiled implementation and compare it with the reference implementation.
   Make sure to compile your code with -O3 optimization level when comparing performance.
#. Try a few different tile sizes and compare the performance.



.. code-block:: cpp

    // Single tile matrix multiplication
    void tile_matmul(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        const int K,
        const int N,
        const int row_offset, // Tile starting indices
        const int col_offset,
        const int k_offset,
        const int TH, // Output tile and A tile height
        const int TW, // Output tile and B tile width
        const int TK // A tile width, B tile height
    ) {
        // Implement multiplication of TH x TK tile by TK x TW matrix to produce TH x TW result.
        // Use offsets to index into matrices A, B and C.
        // Tile in A begins at (row_offset, k_offset) and has size TH x TK.
        // Tile in B begins at (k_offset, col_offset) and has size TK x TW.
        // Resulting tile in C begins at (row_offset, col_offset) and has size TH x TW.
        // Accumulate the result into C (assume it is initialized to 0).
        // Hint: your code will be a lot simpler if you create a helper function to index
        // into a matrix in row-major layout using row and column coordinates and number of columns.
    }

    std::vector<float> tiled_matrix_multiply(
        const std::vector<float>& A,
        const std::vector<float>& B,
        const int M, // Full matrix dimensions
        const int K,
        const int N,
        const int TH, // Output tile and A tile height
        const int TW, // Output tile and B tile width
        const int TK // A tile width, B tile height
    ) {
        // Implement full matrix multiplication using tiling
    }



Introduction to Tenstorrent Architecture
****************************************

Tenstorrent's devices are AI accelerators typically delivered as PCIe cards that can be attached to a standard host.
In this model, the host CPU runs a standard C++ program. In that program, developers can use the TT-Metalium C++ API to configure the accelerator,
allocate memory on the device, and dispatch kernels to Tensix cores. Kernel code is also written in C++.
A high-level view of a Tenstorrent device in the system is shown in Figure 1:

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/tensix_device_on_card.jpg
   :width: 900
   :alt: High-level View of Tensix Device on PCIe Card
   :align: center

   Figure 1: High-level View of Tensix Device on PCIe card

A PCIe card contains one or more Tensix devices, each device consisting of a Tensix processor and a dedicated DRAM.
Note that the device DRAM is separate from the system (host) DRAM, and thus explicit communication is required to transfer data between them.
The Tensix processor contains an array of Tensix cores with a network on-chip (NoC) to pass data from DRAM to fast on-chip SRAM
and between Tensix cores.
On-chip SRAM is often referred to as L1 memory, but it does not operate as a cache; instead, it serves as a working memory
for the Tensix cores.
Each Tensix core has a dedicated compute unit specifically optimized for matrix and vector operations on tensors.

The host program first transfers tensors from host DRAM into the device DRAM.
Tensix cores then move data from device DRAM into on-chip SRAM, perform matrix and vector operations using dedicated hardware units
and may store intermediate results in on-chip SRAM or device DRAM.
Once all the required computation is done, the results are moved back to host memory for further processing, verification, or I/O.

A key design principle of the Tenstorrent architecture is that most computation steps occur entirely on the device,
with explicit and carefully orchestrated data movement.
Typical workloads move input data from the host to device DRAM once, perform many computation steps directly on Tensix cores,
and transfer only final outputs back to the host.
When used this way, host-device communication over PCIe represents a relatively small fraction of the total runtime,
allowing the accelerator's on-device compute and data-movement engines to dominate performance.

In this lab, we will focus on programming a single Tensix core in TT-Metalium. Descriptions of the architectural features
will be limited to this context.
We will extend the architectural and programming model description to multiple Tensix cores in subsequent labs.

Tile-Based Architecture
=======================

The Tenstorrent architecture is a tile-based architecture.
This means that the data is processed in tiles, which are commonly ``32x32`` blocks of data.
Similar to vector architectures, which can perform an operation on one or more vectors of data efficiently,
a Tensix core can perform matrix operations on one or more tiles efficiently.
For example, when performing a matrix multiplication, Tensix core treats each ``32x32`` tile as a single operand
and computes their matrix product as a ``32x32`` result tile, issuing a short sequence of hardware instructions.

Memory Layout and Tiling
------------------------

Most applications deal with data that is larger than a single ``32x32`` tile, so we need to find a way to serve the data to Tensix cores in tile-sized chunks.
In the previous exercise, you performed loop tiling, which changed the memory access pattern without changing the underlying data layout,
which was still stored in row-major order.
An alternative approach is to change the data layout itself by placing all elements of a tile in memory contiguously.
This **tiled layout** is the main memory layout used by the Tenstorrent architecture.

Consider an example of a ``9x4`` matrix. In row-major layout, this matrix is stored in memory as shown in Figure 2:

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/row_major_layout.png
   :width: 600
   :alt: Row-Major Layout of a 9x4 Matrix
   :align: center

   Figure 2: Row-Major Layout of a 9x4 Matrix

Numbers in the matrix in Figure 2 indicate memory addresses that the corresponding element is stored at, not the actual values of the elements.

In tiled memory layout with tile size ``3x2``, this matrix is stored in memory as shown in Figure 3:

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/tiled_layout.png
   :width: 600
   :alt: Tiled Layout of a 9x4 Matrix
   :align: center

   Figure 3: Tiled Layout of a 9x4 Matrix


Once again, numbers in the matrix in Figure 3 indicate memory addresses that the corresponding elements are stored at.
Observe that all elements of a tile are stored contiguously in memory. In this tiled layout there are two "second-order" row-major orderings:

1. Elements within each tile are stored in row-major order.
2. Ordering of tiles relative to other tiles follows row-major ordering. That is, the first tile is stored at the beginning of the memory
   followed by the tile to its right, and so on until the last tile in the row of tiles. Then the next row of tiles is stored,
   starting with the tile in the first column of the next row of tiles.

Note that the number of rows/columns in a matrix may not always evenly divide the number of rows/columns in a tile.
In this case, the matrix needs to be padded with data to the next tile boundary.
This data must not affect the result of the computation, so depending on the operation, it may be padded with zeros or some other value.
In this lab, we will assume that all matrices are aligned to tile boundaries.

With this memory layout, code can now read one tile at a time and find the next tile in memory at a fixed offset rather than having to
assemble a tile by merging different sections of memory, as would be the case for row-major memory layout.
This allows for efficient memory access and computation.


TT-Metalium Programming Model
*****************************

Tenstorrent devices can be programmed at multiple abstraction levels, from high-level neural network libraries to low-level kernel development.
At the highest level, TT-NN provides a PyTorch-like Python API for neural network operations, while TT-Metalium serves as the low-level programming
model that gives developers direct control over the Tensix hardware architecture.

TT-Metalium is unique because it exposes the distinctive architecture of Tenstorrent's Tensix processors.
Unlike traditional GPUs that rely on massive thread parallelism, TT-Metalium enables a cooperative programming model where developers typically
write three types of kernels per Tensix core: reader kernels for data input, compute kernels for calculations, and writer kernels for data output.
These kernels coordinate through circular buffers (CBs) in local SRAM, enabling efficient pipelined execution that overlaps data movement with computation.

TT-Metalium's compute API abstraction layer maintains kernel code compatibility across different hardware generations while ensuring optimal performance.
This means the same kernel code can run efficiently on different Tenstorrent processor generations without requiring developers to rewrite code
for each architecture.
TT-Metalium serves as the foundation for higher-level frameworks, providing the base layer for all Tenstorrent software development.

Before we look at a TT-Metalium program, it is important to emphasize a fundamental distinction between the **host** and the **device**.
The **host** is the conventional CPU system (often x86 or ARM) where the main application runs.
The **device** is the accelerator: a mesh of Tensix cores with their own on-chip memories and access to off-chip DRAM
(recall that this is **device DRAM**, distinct from **host DRAM**).
Tensix cores execute relatively small, specialized programs called kernels that are designed for high-throughput numeric computation,
not for general-purpose tasks.

Crucially, the **host and device live in different address spaces and have different execution models**.
A pointer or object on the host (e.g., an array in system DRAM or a C++ object) is not directly visible to a Tensix core.
To use data on the device, the host must allocate device memory, transfer data into it, and pass data layout information
and device-side memory addresses down to the kernels.
Conversely, when a kernel finishes, any results you want on the host must be explicitly copied back.

There is also a two-stage notion of compile-time vs runtime. The host program is compiled ahead of time, but kernels are
JIT-compiled while the host program runs. At kernel compile time, configuration such as tensor layout and tile sizes are treated
as constants and baked into the binary, allowing aggressive compiler optimization. At kernel launch time, other parameters such as
DRAM base addresses and tile counts are passed as runtime arguments.

Based on this, it may seem that we should specify as many parameters as possible at compile time.
However, in some cases we may choose to pass some information known at compile time as runtime arguments instead.
For example, we may prefer to reuse one kernel binary for many parameter values instead of compiling a new binary for each combination.
Choosing which values to pass at compile time and which to pass at runtime is a trade-off between performance
(more specialization) and flexibility (more reuse).

Understanding which information lives on the host, which is baked into the kernel binary, and which is supplied dynamically at
launch is key to reasoning about performance, correctness, and why the APIs are structured the way they are.


Example TT-Metalium Program
***************************

We now present a simple example TT-Metalium program that performs an elementwise addition of two tensors of shape ``MxN``.
This program will be used to illustrate the TT-Metalium programming model, different types of kernels, and how they map to the underlying architecture.
Key points will be highlighted in this text. Detailed comments are provided in the C++ code to help with code understanding.

Exercise 2: Running the Example Program
=======================================

If you haven't already done so, clone an appropriate release of the TT-Metalium repository from https://github.com/tenstorrent/tt-metal
Make sure you are in the ``tt-metal`` directory and then build the example program, using the following commands:

.. code-block:: bash

   export TT_METAL_HOME=$PWD
   ./build_metal.sh
   ./build/ttnn/examples/example_lab_eltwise_binary

Make sure that the program executes correctly and that the output says "Test Passed" on the host terminal.

Program Description
===================

The main program for the code example being discussed is located in the file ``ttnn/examples/lab_eltwise_binary/lab_eltwise_binary.cpp``.
The first thing to emphasize is that all the code in this file executes on the host, although there are many API calls that cause activity on the device.

Looking at the main function, we see that the host program first initializes input data for the operation and performs a reference computation on the host CPU.
This will be used to verify the correctness of the TT-Metalium implementation. Note that the data type used is bfloat16 (brain floating-point), which is a
16-bit floating-point format commonly used in AI applications. Since the host CPU doesn't natively support bfloat16,
we use the ``bfloat16`` class from the ``tt-metal`` library and cast data between this type and single-precision (32-bit) floating-point as needed.

Next, the host program initializes the Tensix device and program state by calling the function ``init_program``.
This function contains a lot of boilerplate code that configures the device to run our code on a single Tensix core. Most programs utilizing
a Tensix device would use similar code to configure the device and program, and exact initialization details are not important for this lab.

After initialization, the program calls the function ``eltwise_add_tensix``, which is the main function that configures and creates kernels
and triggers elementwise addition on the Tensix device.
Finally, the program validates the results by comparing the Tensix output with the reference computation on the host CPU.


Kernel Types and Data Flow
--------------------------

Before diving into the function ``eltwise_add_tensix``, let us discuss the different types of kernels and how they map to the underlying hardware.
Programming with Metalium typically requires three kernel types per Tensix core: a **reader kernel** for data input,
a **compute kernel** for calculations, and a **writer kernel** for data output.
Reader and writer kernels are collectively referred to as data movement kernels.
Data movement and compute kernels communicate through circular buffers (CBs) in internal SRAM.
The circular buffers act as producer-consumer FIFO (First In First Out) queues, enabling safe and efficient data exchange between kernels.
A kernel can even read from and write to the same circular buffer, allowing processing to be divided into multiple
steps, all of which use uniform interfaces (CBs) to promote code reuse.
Each circular buffer is assumed to have only one reader kernel and one writer kernel.
Note that the circular buffers typically contain only a small number of tiles at a time, not the entire tensor.

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/cb_data_flow.jpg
   :width: 900
   :alt: Kernel Data Flow Through Circular Buffers
   :align: center

   Figure 4: Kernel Data Flow Through Circular Buffers

Each kernel interacts with the buffers as follows:

- **Reader kernel:** Reads data (e.g. from device DRAM) into the circular buffers and signals when new data has been read and is available.
- **Compute kernel:** Waits for data to become available in its input circular buffers before processing it. After computation, it writes the results to
  one or more output circular buffers and marks data as ready.
- **Writer kernel:** Waits for the computed results to appear in the buffers before writing them to the output location (e.g. device DRAM).

This mechanism ensures that each kernel only proceeds when the necessary data is ready, preventing race conditions and enabling asynchronous,
pipelined execution across the hardware. Different kernel types are mapped to the Tensix core, whose high-level diagram is shown in Figure 5.

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/tensix_core.png
   :width: 600
   :alt: Top-Level Diagram of Tensix Core
   :align: center

   Figure 5: Top-Level Diagram of Tensix Core

The Tensix core consists of four major parts:

1. Internal SRAM (L1) memory - Stores input/output tiles in circular buffers for fast access by the Tensix Engine.
   It also holds program code for all RISC-V processors within the core.
2. Two routers - Manage data movement between internal SRAM (L1) memory, device DRAM, and other Tensix cores.
3. Tensix Engine - Hardware accelerator that efficiently performs matrix and vector computations on tiles.
4. Five RISC-V Processors that control the Tensix Engine and routers:

   - RISC-V 0 and RISC-V 4 - These processors control routers to exchange data between the Internal SRAM and device DRAM (or other Tensix cores).
     Either of these can be used for a reader or writer kernel.
   - RISC-V 1 through RISC-V 3 - These processors control the Tensix Engine through specialized Tensix instructions.
     Note that these RISC-V processors don't perform actual tile computations.
     Instead, they serve as microcontrollers directing the operations of the Tensix Engine.
     One RISC-V processor is responsible for issuing commands to the compute engine, while the other two
     are responsible for transferring tile data between circular buffers in SRAM and Tensix Engine registers.
     Compute kernel code defines functionality for all three of these processors.


Kernel Creation and Configuration
---------------------------------

The function ``eltwise_add_tensix`` creates and configures the kernels that will be used to perform the elementwise addition on the Tensix device.
At a high level, the function creates a tensor object that resides in device DRAM and then creates two dataflow kernels, one reader and one writer,
one compute kernel, and three circular buffers to pass data between the kernels, and then triggers kernel execution on the device.

The function creates three Tensor objects of shape ``MxN`` using the tile layout described earlier.
These tensors are allocated in device DRAM, which is distinct from host DRAM and is directly attached to the Tensix processor.
The input tensors are created and initialized by transferring the data from the host to the device in one step using ``Tensor::from_vector``.
The vectors passed to ``Tensor::from_vector`` are the same input vectors that were used for the reference computation.
Because the ``TensorSpec`` object passed to ``Tensor::from_vector`` specifies the tile layout, the data is automatically organized in a tiled
memory layout when stored on the device. This is desirable because the matrix engine is optimized for operations on tiled data.

The function then creates three circular buffers to enable data movement between kernels.
A circular buffer is a FIFO buffer with configurable size.
Creating a circular buffer simply means allocating sufficient device SRAM memory based on the specified configuration, and associating
the specified circular buffer index with this SRAM memory and its configuration.
In our example program, circular buffers are created with two tiles each to allow for double buffering. For example, a reader kernel
can be reading one tile while the compute kernel is processing the other tile, enabling pipelined execution.
The number of tiles in a circular buffer can be adjusted to trade off memory for performance, but generally there are diminishing
returns beyond a few tiles.

The function creates the three types of kernels discussed earlier: reader, compute, and writer.
Creating a kernel registers it with the program object, so that it can be executed later.
Each kernel can take two types of arguments: compile-time and runtime kernel arguments, as mentioned earlier.

.. _kernel-args-blurb:

The two types of kernel arguments differ in *when* their values are determined and *how* they are used.

**Compile-time kernel arguments**

* Values that are known when the kernel is built (JIT-compiled).

* They are hard-coded into the kernel binary and can be used by the compiler to specialize and optimize
  the code, by e.g. unrolling loops, removing branches, choosing specific data paths, etc.

* Changing a compile-time argument effectively means generating a new version of the kernel binary.

* These arguments are provided by the host during kernel configuration as ``compile_args``

**Runtime kernel arguments**

* Values that are provided by the host each time the kernel is launched, possibly varying per core.

* Stored in a small argument block in the core's local memory and read by the kernel at the start of execution, for example using ``get_arg_val<T>(index)``.

* They do not change the compiled binary; instead, they affect kernel behavior at runtime. Examples include buffer base addresses,
  flags to enable/disable certain features, etc.

* Same compiled kernel can be reused many times with different runtime arguments, without recompiling.

In summary, compile-time arguments specialize the kernel *code itself*, while runtime arguments specialize *what that code does on a particular launch*.

In our example program, reader and writer kernels take information about tensor layout and data distribution as compile-time arguments.
Compile-time arguments are passed as a vector of ``uint32_t`` values. The ``TensorAccessorArgs`` utility is a clean way to append relevant tensor layout
information into this ``uint32_t`` vector, without the programmer having to worry about internal details.

Dataflow kernels take the base addresses of the input and output buffers in device DRAM, along with the number of tiles to process
as runtime arguments.

The compute kernel takes the number of tiles to process as a compile-time argument and doesn't take any runtime arguments.
At first, it may seem like an odd choice to pass the number of tiles as a compile-time argument to the compute kernel,
but as a runtime argument to dataflow kernels.
Since using compile-time arguments enables various compiler optimizations, it is particularly suitable to use them for compute kernels, which are compute bound.
While passing the number of tiles as a compile-time argument to dataflow kernels would also work, they may not benefit much since their performance is memory bound.
At the same time, using compile-time arguments for dataflow kernels would cause them to be recompiled for each different number of tiles.
Since the performance benefit of using compile-time arguments is not significant for dataflow kernels,
we optimize for code reuse and avoid recompilation by using runtime arguments for dataflow kernels.
This distinction will become more apparent in subsequent labs when we start working with multiple Tensix cores.

Finally, the function executes the kernels by adding the program to the workload and enqueuing it for execution, which triggers kernel JIT compilation
followed by kernel execution on the device. It is useful to remind ourselves that until this point, all the code we discussed executed on the host,
not on the device. We will examine kernel code next.

Reader Kernel Code
------------------

The function can be summarized by the following pseudo-code:

.. code-block:: cpp

   read_runtime_arguments()
   read_compile_time_arguments()
   create_address_generators()
   for (i in 0 .. n_tiles) {
       transfer_tile_from_dram_to_circular_buffer(in0, i)
       transfer_tile_from_dram_to_circular_buffer(in1, i)
   }

The reader kernel in ``ttnn/examples/lab_eltwise_binary/kernels/dataflow/read_tiles.cpp`` is responsible for transferring data
from device DRAM into circular buffers located in internal device SRAM, where it can be efficiently accessed by the compute kernel.
The kernel reads the base addresses of the two input tensors in DRAM and the total number of tiles to process as runtime arguments.

The kernel uses two circular buffers (``c_0`` and ``c_1``) as destination buffers for the two input tensors.
It retrieves the tile size from the circular buffer configuration, which must match the tile size used in the DRAM buffers.

Recall that the host uses ``TensorAccessorArgs`` to pack tensor shape and layout into a compile-time ``uint32_t`` argument vector.
On the device, this vector is unpacked into a device-side ``TensorAccessorArgs`` (a different underlying type but same name),
which is then combined with the runtime base addresses to create an address generator object (``TensorAccessor``).
An address generator object abstracts away the complexity of physical memory layout, such as data distribution among DRAM banks,
by automatically computing the physical DRAM address for any given tile index.

The main processing loop iterates over all tiles, implementing a producer-consumer pattern with the compute kernel.
For each tile, the kernel first reserves space in both circular buffers using blocking calls to ``cb_reserve_back``
to ensure that space is available before attempting to write.
Once space is reserved, the kernel obtains write pointers to the circular buffers and initiates two non-blocking asynchronous
read operations using ``noc_async_read_tile``. Observe that this call takes in the index of the tile being read and an address generator,
along with the circular buffer address to write the tile to. The address generator automatically maps the logical tile index to the correct physical
DRAM address based on the specific memory layout.

Because ``noc_async_read_tile`` is non-blocking, the two reads can proceed in parallel if sufficient bandwidth is available, transferring data from DRAM to
the circular buffers simultaneously. After both reads are initiated, the kernel calls ``noc_async_read_barrier`` to wait
for both transfers to complete. This is important because the kernel should not signal that the tiles are ready for consumption
until data is actually available.

The reader kernel repeats this process for all tiles. Given that circular buffers were created with two tiles each, the reader kernel can
read a new tile while the compute kernel is processing the previous one.

Compute Kernel Code
-------------------

The compute kernel in ``ttnn/examples/lab_eltwise_binary/kernels/compute/tiles_add.cpp`` is responsible for performing the elementwise addition of two tiles.

The function can be summarized by the following pseudo-code:

.. code-block:: cpp

   read_compile_time_arguments()
   initialize_tensix_engine_for_elementwise_addition()
   for (i in 0 .. n_tiles) {
       add_tiles_in_input_circular_buffers()
       write_result_to_output_circular_buffer()
   }

The kernel reads the number of tiles to process as a compile-time argument, enabling compiler optimizations such as
loop unrolling.

An important architectural detail is that the compute kernel actually runs on three different RISC-V processors
within the Tensix core: an unpacker (RISC-V 1 in Figure 5), a compute processor (RISC-V 2), and a packer (RISC-V 3).
The compiler automatically generates appropriate code for each of these three processors from the same source code,
relieving the programmer from having to write different code for each processor.
The unpacker controls reading data from circular buffers, the compute processor issues the actual arithmetic operations
using the Floating-Point Unit (FPU) of the Tensix Engine, and the packer controls writing results back to circular buffers.
It is worth repeating that these RISC-V processors don't perform actual computations or packing/unpacking operations.
They simply issue commands to the Tensix Engine to perform the actual computations and packing/unpacking operations.

The compute kernel uses circular buffers ``c_0`` and ``c_1`` for the two input tensors and ``c_16`` for the output tensor.
There are 32 circular buffers in total (0-31), and the exact indices used are up to programmer's discretion, provided they are used consistently
(i.e. reader and writer kernels must use the corresponding indices as the compute kernel).

The kernel initializes the Tensix Engine for elementwise binary operations, first calling ``binary_op_init_common``
to set up the general binary operation hardware infrastructure, followed by ``add_tiles_init`` to configure the FPU
specifically for addition.
This initialization only needs to be done once before the main loop, since all tiles use the same operation.

The main processing loop iterates over all tiles.
For each tile, the kernel first waits for one tile to become available in each input circular buffer using
blocking calls to ``cb_wait_front``.
These blocking calls ensure that the compute kernel doesn't attempt to use the data before the reader kernel
has finished transferring it.
The compute kernel then acquires access to the destination register array using ``tile_regs_acquire``
and calls ``add_tiles`` to perform the elementwise addition.
The destination register array is a special storage area in the Tensix Engine that can hold multiple tiles
and serves as the temporary output location for FPU computations. The acquire operation also
initializes all tiles in the destination register array to zero, which is not important for
this example program, but is useful for operations like matrix multiplication where results accumulate.

After the computation completes, the kernel marks the input tiles as consumed using ``cb_pop_front``
to free space in the circular buffers, then releases the destination register using ``tile_regs_commit``
to signal that the compute core has finished writing, which allows the packer to read the result.

The packer core for its part waits for the destination register to be ready using ``tile_regs_wait``,
ensures there is space in the output circular buffer using ``cb_reserve_back``, and then copies the
result from the destination register to the output circular buffer using ``pack_tile``.
Finally, it marks the output tile as ready using ``cb_push_back`` and releases the destination
register using ``tile_regs_release``.

While it may seem like some of these operations are redundant (e.g. waiting on the destination register
when it has seemingly just been released), it is important to remember that compute kernel code is executed
on three different RISC-V processors. This synchronization mechanism using acquire, commit, wait,
and release ensures that the three RISC processors coordinate properly, with the compute processor
writing results to the intermediate destination register array, and the packer processor reading
them from it without conflicts.


Writer Kernel Code
------------------

The writer kernel in ``ttnn/examples/lab_eltwise_binary/kernels/dataflow/write_tiles.cpp`` is responsible for transferring
computed results from the circular buffer in internal device SRAM back to device DRAM.
The kernel code can be summarized by the following pseudo-code:

.. code-block:: cpp

   for (i in 0 .. n_tiles) {
       transfer_tile_from_circular_buffer_to_dram(out0, i)
   }

Most of the code is similar to the reader kernel, with the main difference being that the writer kernel writes to DRAM instead of reading from it.
The circular buffer the writer kernel reads from has capacity for two tiles, allowing the compute kernel to write to one new tile, while the writer
kernel is reading from the previously produced tile.
This coordination between the compute and writer kernels enables pipelined execution, where computation and data movement can overlap.


Example Program Summary
=======================

It is useful to wrap up this example description by emphasizing one more time the nature of the
TT-Metalium programming model and division of tasks and data between host and device.
At a high level, all kernel code (``read_tiles.cpp``, ``write_tiles.cpp``, ``tiles_add.cpp``)
executes on the device, and all its C++ objects are created on the device.
Specifically:

* Ordinary local variables in kernel code are stored either in local SRAM or in RISC-V registers.

* Tensor data is stored in device DRAM, which is directly attached to the Tensix processor.

* Circular buffers are implemented in fast on-chip device SRAM and are used to store tiles of data,
  containing a subset of the data in a tensor or intermediate results of computations.

Conversely, all code in ``lab_eltwise_binary.cpp`` executes on the host, and all its C++ objects are created on the host
(either in CPU registers or host DRAM). Obvious examples include vectors of data and various local variables.
However, some host-side objects contain information about data and code on the device. Specifically:

* ``Tensor`` objects are created on the host and contain information about the tensor shape and layout,
  but actual tensor data is stored in device DRAM.

* ``TensorAccessorArgs`` objects exist on both host and device, but they are different underlying types,
  defined in different headers (``tt_metal/api/tt-metalium/tensor_accessor_args.hpp`` for host-side
  and ``tt_metal/hw/inc/api/tensor/tensor_accessor_args.h`` for device-side).

* Integers ``src0_addr``, ``src1_addr``, and ``dst_addr`` in the ``eltwise_add_tensix()`` function
  are host-side integers, but they contain addresses of the input and output tensor data in device DRAM.

* Kernel code and their arguments are JIT-compiled on the host, but then transferred to the device for execution.

Understanding the location of data and code on the host and device is useful when debugging or analyzing performance.


Kernel Compilation and Execution
********************************

As mentioned earlier, kernels are JIT compiled and executed on the device. This presents both advantages and disadvantages during development.
On the one hand, if one updates only kernel code, there is no need to rebuild before running the program to test that the changes had the desired effect.
On the other hand, it also means that errors in the kernel code will not be caught at host-code compile time, but only at time of host code execution,
when JIT compilation is triggered.

Exercise 3: Observing JIT Compile Errors
========================================

Perform the following steps:

#. Introduce a syntax error in the reader kernel.

#. Rebuild the example program by running ``./build_metal.sh`` and observe that no error is reported.

#. Run the example program by running ``./build/ttnn/examples/example_lab_eltwise_binary`` and
   observe how JIT compilation errors are reported.

#. Fix the syntax error and rerun the program to confirm that the program now runs correctly with **no rebuilding step required**.


Debug Facilities in TT-Metalium
*******************************

Host code can be debugged using the usual debugger tools like ``gdb``.
To debug host code, build the program with debug symbols:

.. code-block:: bash

   ./build_metal.sh --build-type Debug

Then run the program using ``gdb``:

.. code-block:: bash

   gdb ./build/ttnn/examples/example_lab_eltwise_binary

Kernels cannot be easily debugged using ``gdb``, and TT-Metalium provides a number of other methods for debugging kernels.
These methods are useful for debugging hangs and other issues that may not be apparent from the host-side code.


Debug Print API
===============

Because kernel code runs on the device, it can't use standard C++ functions to print debug information,
as the device doesn't have a terminal.
The Debug Print (DPRINT) API is a device-side debugging feature that lets a kernel print values back
to the host while it runs. You can think of it as a constrained, lightweight alternative to ``printf`` that works inside kernels.
It is mainly used to inspect scalar variables, addresses, and the contents of tiles stored in circular buffers, which helps when debugging
numerical issues or hangs.

The DPRINT API is controlled through environment variables on the host side.
The host-side environment variable ``TT_METAL_DPRINT_CORES`` specifies which cores' DPRINT information will be forwarded to the host terminal.
If this environment variable is not set, calls to DPRINT APIs produce no output and behave as no-ops at runtime.
When debugging, it is common to set this variable to ``all`` to print from all cores (i.e. ``export TT_METAL_DPRINT_CORES=all``).
When not debugging, you should unset this variable to disable printing (i.e. ``unset TT_METAL_DPRINT_CORES``).
Unsetting this variable is particularly important to do when evaluating performance.

An alternative to printing to the host terminal is to print to a log file, which can be done by setting
the ``TT_METAL_DPRINT_FILE`` environment variable (e.g. ``export TT_METAL_DPRINT_FILE=log.txt``).

To use DPRINT in a kernel, you include the debug header and use a C++ stream-like syntax:
DPRINT supports printing integers, floats, and strings, but it does **not** directly support the C++ ``bool`` type.
The common pattern is to cast a Boolean to an integer (for example, ``uint32_t``) before printing it.
The following example shows a simple print of local kernel variables, including a Boolean flag:

.. code-block:: cpp

   #include "api/debug/dprint.h"

   void kernel_main() {
       uint32_t iter = 5;
       bool done = false;

       // DPRINT uses a streaming syntax; ENDL() flushes the print buffer.
       DPRINT << "iter = " << iter << ENDL();

       // Booleans should be cast to an integer type before printing.
       DPRINT << "done flag = " << static_cast<uint32_t>(done) << ENDL();
   }

Printing a Tile in a Dataflow Kernel
------------------------------------

Data is passed between kernels using circular buffers (CBs), which often contain tiles of data.
DPRINT can be combined with the ``TileSlice`` helper to print part or all of a tile from a CB.

You can only safely sample a tile from a CB **between** the appropriate CB API calls:

- When reading from CBs (e.g. in writer kernels): between ``cb_wait_front()`` and ``cb_pop_front()``.
- When writing to CBs (e.g. in reader kernels): between ``cb_reserve_back()`` and ``cb_push_back()``.

A simplified example of printing a full tile from an output CB in a writer kernel is shown below.

.. code-block:: cpp

   #include "api/dataflow/dataflow_api.h"
   #include "api/debug/dprint.h"
   #include "api/debug/dprint_tile.h"
   #include "tt-metalium/constants.hpp"

   void kernel_main() {
       // Assume this circular buffer holds output tiles from a compute kernel.
       constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

       // Number of tiles to consume and optionally print.
       uint32_t n_tiles = get_arg_val<uint32_t>(0);

       for (uint32_t t = 0; t < n_tiles; ++t) {
           // Wait until one tile is available at the front of cb_out.
           cb_wait_front(cb_out, 1);

           DPRINT << "Output tile " << t << " from cb_out = " << static_cast<uint32_t>(cb_out) << ENDL();

           // Print each row of a tile.
           // TileSlice has limited capacity, so we must print one row at a time.
           for (uint32_t row = 0; row < tt::constants::TILE_HEIGHT; row++) {
               SliceRange slice_range {
                   .h0 = static_cast<uint8_t>(row),
                   .h1 = static_cast<uint8_t>(row + 1), // One row
                   .hs = 1,                             // Stride is 1
                   .w0 = 0,
                   .w1 = tt::constants::TILE_WIDTH,     // Full width
                   .ws = 1                              // Stride is 1
               };

               // TileSlice(cb_id, tile_idx, slice_range, cb_type, ptr_type)
               DPRINT << row << ": "
                      << TileSlice(
                         static_cast<uint8_t>(cb_out),
                         /* tile_idx = */ 0,
                         slice_range,
                        /* cb_type = */ TSLICE_OUTPUT_CB,
                        /* ptr_type = */ TSLICE_RD_PTR)
                      << ENDL();
           }

           uint32_t cb_out_addr = get_read_ptr(cb_out);

           // Perform the actual work of this writer kernel...

           // Mark this tile as consumed in the CB.
           cb_pop_front(cb_out, 1);
       }
   }

The ``cb_type`` parameter to ``TileSlice`` specifies whether the CB is the input or output of the compute engine.
The ``ptr_type`` parameter specifies whether we wish to get data from the read-side (front), or the write-side (back) of the CB.
In writer kernels, these are most commonly set to ``TSLICE_OUTPUT_CB`` and ``TSLICE_RD_PTR``, respectively,
to access the data that is about to be read from the CB.
In reader kernels, these are most commonly set to ``TSLICE_INPUT_CB`` and ``TSLICE_WR_PTR``, respectively,
to access the data that has just been written (but not yet "pushed back") to the CB.


Caveats and Best Practices
--------------------------

A few important caveats to keep in mind when using DPRINT:

- **Flushing behavior**

  DPRINT output is only guaranteed to flush when you print ``ENDL()`` or ``'\n'``, or when the device closes.
  Always end each logical debug line with ``ENDL()`` if you want to see it promptly.

- **Kernel size and string length**

  Each distinct DPRINT call embeds a format string (and often the file name and line number) into the kernel binary.
  Long or numerous debug strings increase kernel size and may cause it to not fit into available internal SRAM.
  To avoid this, keep DPRINT messages short, particularly if printing within a loop with many iterations.
  You can also reduce the number of iterations by reducing the problem size while debugging.
  Finally, remove or disable most DPRINTs once you have diagnosed the issue.

Taken together, these practices let you use DPRINT as a practical, low-level debug tool in TT-Metalium kernels without
needing deep knowledge of the underlying Tenstorrent architecture, while still avoiding common pitfalls.


Exercise 4: Using DPRINT to Debug a Kernel
------------------------------------------

Add DPRINT statements to the writer kernel in our example program to print:

* Value of the iterator ``i`` in every iteration of the ``for`` loop
* Contents of the resulting tile for the first three tiles processed by the kernel.

For testing purposes, modify the program's input data to not use random numbers
so you can verify that the results are as expected. Keep in mind that the input data vector is in row-major order,
but it is then stored in tiled layout in the tensor.
Also keep in mind that ``bfloat16`` has limited precision, so you may run into seemingly unexpected results if you
perform a naive operation like ``x = x + 0.1`` inside of a loop if the result rounds to the original value ``x``.
Note that this is a common issue with floating-point arithmetic, and is not specific to ``bfloat16``, but you are
more likely to encounter it with ``bfloat16`` because of its limited precision.
Similarly, many integers cannot be represented exactly as floating-point numbers, and this becomes apparent
much sooner with lower precision types, such as ``bfloat16``.

It is also worth noting that printing individual ``bfloat16`` values requires casting the value to a ``float``
to get expected floating-point result (e.g. ``std::cout << "x: " << static_cast<float>(x);``).
``TileSlice`` takes care of this internally, so no further casting is needed when printing tiles.

Since this exercise will involve modifying the host-side code, you will need to rebuild the program before rerunning it.
The easiest way to rebuild the program is to rerun ``./build_metal.sh`` from the ``tt-metal`` directory.

Debugging Hangs using Stack Traces
==================================

TT-Metalium includes a Python tool called ``tt-triage``, which inspects a hung TT-Metalium run and prints per-core
call stacks for the RISC-V processors on all cores. This is often the fastest way to see exactly where the device got stuck.
When a program hangs, **leave it running**, open another terminal and run the following command from the ``tt-metal`` directory:

.. code-block:: bash

   python tools/triage/dump_callstacks.py

This will print the call stacks for all RISC-V processors on all cores.

``tt-triage`` Dependencies
--------------------------

Depending on your environment, you may encounter an error running the above command, such as
``Module 'No module named 'ttexalens'' not found. Please install tt-exalens``, or
``Debugger version mismatch``.
If this occurs, you can address it by creating a Python virtual environment and installing dependencies,
by running the following from the ``tt-metal`` directory:

.. code-block:: bash

   ./create_venv.sh
   source python_env/bin/activate
   scripts/install_debugger.sh
   pip install -r tools/triage/requirements.txt

Note that you may need to reenter the virtual environment by re-running ``source python_env/bin/activate``
if you open a new terminal later.


Exercise 5: Using tt-triage to Debug a Hang
-------------------------------------------

To illustrate how ``tt-triage`` can be used to debug a hang, we will use the ``lab_eltwise_binary`` example program.
You can introduce a very simple artificial hang by commenting out the calls to ``cb_pop_front``
(as if you accidentally forgot them) in the compute kernel in
``ttnn/examples/lab_eltwise_binary/kernels/compute/tiles_add.cpp``.

With this change, for the first two tiles, the program should run normally, but then the reader kernel blocks waiting
for space in the circular buffers, which from the host side looks like a hang.
To help pinpoint the problem, you can dump stack traces.

#. Make changes to the compute kernel as suggested above.

#. Run the program as usual.

#. When it becomes apparent that the program has been running for a long time without indication of progress,
   keep it running and open another terminal and run ``python tools/triage/dump_callstacks.py`` from the ``tt-metal`` directory.

The output will show the call stacks for all RISC-V processors on all cores, including cores
that are running firmware responsible for dispatching kernel code. You should ignore the cores that are running firmware,
and focus on the cores that are running kernel code. In our example, these will be in location ``(0,0)``, since that is
the core we specified in ``init_program()`` in ``ttnn/examples/lab_eltwise_binary/lab_eltwise_binary.cpp``.
Another way to recognize relevant RISC-V processors is to look under the **Kernel Name** column, and
look for kernel names of interest, such as ``read_tiles`` or ``write_tiles``.

In this case, you should see the call stack for the ``read_tiles`` kernel contains a call to ``cb_reserve_back``, even if you dump stack trace
repeatedly, indicating a possible source of the problem.
In general, stack traces alone may not be sufficient to uncover the reason for the hang. In such a case,
you may need to add DPRINT statements to kernel code to help pinpoint the problem. For example,
printing iterator values in all kernels may be useful to identify the iteration when the hang occurs.

Note that you can terminate the hung program by pressing ``Ctrl`` + ``C`` in the terminal where it is running.
Once you are done debugging, uncomment the calls to ``cb_pop_front`` in the compute kernel to restore normal behavior.

Device Performance Profiling
============================

TT-Metalium includes a device program profiler that measures how long sections of your device kernels take to run.
Profiling is disabled by default, but can be enabled by setting the ``TT_METAL_DEVICE_PROFILER`` environment variable to ``1``
when launching the binary.
With this flag set, when the program finishes and the device is closed (e.g. via ``mesh_device->close()``), the runtime
automatically pulls the profiling data from the device and writes it to a CSV log file on the host.
The CSV file is named ``profile_log_device.csv`` and is stored in the ``generated/profiler/.logs/`` directory.
The log file contains device-side profiling data for all RISC-V processors on all cores, tagged as
``*RISC-KERNEL`` and ``*RISC-FW``, corresponding to kernel execution time and overall firmware (kernel + runtime support) execution time.
Note that the log file uses names BRISC and NCRISC for the two RISC-V processors that control routers
(RISC-V 0 and RISC-V 4 in Figure 5), and TRISC for the remaining Tensix RISC-V processors (RISC-V 1 through RISC-V 3 in Figure 5).

Exercise 6: Using Device Profiling to Profile Kernels
-----------------------------------------------------

#. **Make sure code is built with Release option**

   If you previously used the ``--build-type Debug`` flag, do not forget to rebuild the programming examples
   with the ``--build-type Release`` flag before profiling performance.

#. **Make sure DPRINTs are disabled**

   Ensure that the ``TT_METAL_DPRINT_CORES`` environment variable is not set.
   The device profiler and kernel DPRINT both consume the same limited on-chip SRAM,
   so only one should be enabled at a time. Also, the time spent on DPRINT may affect performance measurements,
   leading to misleading results when compared to another run without DPRINT.

#. **Run the program with profiler enabled**

   .. code-block:: bash

      TT_METAL_DEVICE_PROFILER=1 ./build/ttnn/examples/example_lab_eltwise_binary

#. **Locate the CSV log file**

   Once the run finishes, locate the CSV log file and open it in a spreadsheet editor such as Excel, or a text editor.
   The file contains one row per profiling event. It begins with a header line that includes the chip frequency, for example:

   .. code-block:: text

      ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120
      PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, trace id, trace id counter, zone name, type, source line, source file, meta data

#. **Compute elapsed firmware time**

   In the CSV log file, each RISC-V processor on each core has several rows of data, each indicating a unique timer event.
   Column ``time[cycles since reset]`` indicates the number of cycles since the reset of the device until the specific timer event.
   For this lab, it is sufficient to determine the overall firmware execution time. To compute it,
   simply subtract the maximum and minimum ``time[cycles since reset]`` values across all rows in the log file, and then multiply
   the difference by the clock cycle time, which can be calculated from the chip frequency in the header.
   The time computed this way is the total elapsed time in the firmware and does not include any host execution time,
   or data transfer from the host to the device.


Matrix Multiplication in TT-Metalium
************************************

Now that you have a basic understanding of using the TT-Metalium APIs and building data movement and compute kernels,
we can look at a more complex example of matrix multiplication.
As described earlier, matrix multiplication can be decomposed into a series of tile multiplications via tiling.
In Exercise 1, you implemented a tiled version of matrix multiplication by identifying appropriate indices in the input and output matrices,
which were organized in a row-major layout, and then performing the necessary arithmetic operations on these smaller tiles.
We will now show how the same can be achieved in TT-Metalium, while taking advantage of the built-in tiled memory layout.

The key insight is that tiled matrix multiplication can be performed by considering each tile as an element of a larger matrix,
and then performing regular matrix multiplication on these larger matrices, where each tile is multiplied by another tile
using standard matrix multiplication.
While we will not present a formal proof of correctness, we will illustrate how this works intuitively to allow us to
write correct kernel code to perform this operation.

Consider multiplication of an ``MxK`` matrix ``A`` and a ``KxN`` matrix ``B``. The ``C[i, j]`` element of the resulting matrix is computed as
the dot product of row ``i`` of ``A`` with column ``j`` of ``B``. The dot product is computed by multiplying corresponding pairs of elements
and summing them. Extending this idea to two neighboring elements of ``C``, say ``C[i, j]`` and ``C[i + 1, j]``, we need rows ``i``
and ``i + 1`` of ``A`` and the same column ``j`` of ``B``.
More generally, if we want to compute a rectangular tile of height ``TILE_HEIGHT`` and width ``TILE_WIDTH`` starting at ``C[i, j]``,
we must fetch the entire band of tiles in ``A`` covering rows ``i`` to ``i + TILE_HEIGHT - 1`` (across all columns),
and the entire band of tiles in ``B`` covering columns ``j`` to ``j + TILE_WIDTH - 1`` (across all rows).
Conceptually, we are treating the ``K`` dimension as being split into tile-sized chunks, and for each output tile we
accumulate products over all those K-tiles.

Consider the concrete example shown in Figure 6.

.. figure:: https://github.com/tenstorrent/tutorial-assets/blob/main/media/tt_metal/labs/lab1/tiled_matrix_mul_example.png
   :width: 1200
   :alt: Tiled Matrix Multiplication Example
   :align: center

   Figure 6: Tiled Matrix Multiplication Example

Figure 6 shows an example where ``A`` is a ``9x4`` matrix, and ``B`` is a ``4x6`` matrix.
If we choose ``3x3`` tiles for matrix ``C``, we need 3 rows of matrix ``A`` and 3 columns of matrix ``B`` to compute a single tile of matrix ``C``.
This means that ``A`` tiles must have 3 rows, ``B`` tiles must have 3 columns, and
the inner tile dimensions must match (the number of columns in an ``A`` tile equals
the number of rows in a ``B`` tile).
If we choose ``3x2`` tiles for matrix ``A``, we can divide ``A`` into six tiles ``A0`` through ``A5``.
The figure shows labeling of the tiles in row-major order, which is how tiled layout works on the Tenstorrent architecture, as described earlier.
We can similarly divide ``B`` into four tiles ``B0`` through ``B3``, each of shape ``2x3``.
Each ``C`` tile is computed by summing products of the corresponding tile row of ``A`` and tile column of ``B``, exactly like scalar matrix multiplication,
but with tiles instead of individual numbers. For instance, tile ``C0`` corresponds to tile row 0 of ``A`` and tile column 0 of ``B``, and therefore

``C0 = A0 * B0 + A1 * B2``

Each product in this equation is an inner ``3x2`` by ``2x3`` matrix multiplication producing a ``3x3`` tile that is accumulated into ``C0``.
We can summarize computations for all ``C`` tiles in a table as follows:

+-----------+--------+--------+-----------+--------+--------+
| Computing | From A | From B |   ``+``   | From A | From B |
+-----------+--------+--------+-----------+--------+--------+
| C0        | A0     | B0     |   ``+``   | A1     | B2     |
+-----------+--------+--------+-----------+--------+--------+
| C1        | A0     | B1     |   ``+``   | A1     | B3     |
+-----------+--------+--------+-----------+--------+--------+
| C2        | A2     | B0     |   ``+``   | A3     | B2     |
+-----------+--------+--------+-----------+--------+--------+
| C3        | A2     | B1     |   ``+``   | A3     | B3     |
+-----------+--------+--------+-----------+--------+--------+
| C4        | A4     | B0     |   ``+``   | A5     | B2     |
+-----------+--------+--------+-----------+--------+--------+
| C5        | A4     | B1     |   ``+``   | A5     | B3     |
+-----------+--------+--------+-----------+--------+--------+

Further splitting each row so that only one multiplication is performed in each step, we get the following table:

+-----------+--------+--------+
| Computing | From A | From B |
+-----------+--------+--------+
| C0        | A0     | B0     |
+           +--------+--------+
|           | A1     | B2     |
+-----------+--------+--------+
| C1        | A0     | B1     |
+           +--------+--------+
|           | A1     | B3     |
+-----------+--------+--------+
| C2        | A2     | B0     |
+           +--------+--------+
|           | A3     | B2     |
+-----------+--------+--------+
| C3        | A2     | B1     |
+           +--------+--------+
|           | A3     | B3     |
+-----------+--------+--------+
| C4        | A4     | B0     |
+           +--------+--------+
|           | A5     | B2     |
+-----------+--------+--------+
| C5        | A4     | B1     |
+           +--------+--------+
|           | A5     | B3     |
+-----------+--------+--------+


From this table, we can observe that if we compute the ``C`` tiles in row-major order (``C0``, ``C1``, ``C2``, ..., ``C5``),
we will visit tiles of ``A`` and ``B`` in a regular pattern, visiting one row of tiles of ``A`` with all columns of tiles of ``B``.
For example, to compute ``C0`` and ``C1`` we start with the row of tiles ``A0``, ``A1`` while cycling through columns of tiles ``B0``, ``B2`` followed by ``B1``, ``B3``.
From this viewpoint, we can think of ``A0``, ``A1``, ..., ``A5`` as the "elements" of a ``3x2`` tile matrix, ``B0``, ..., ``B3`` as the "elements" of a ``2x2`` tile matrix,
and ``C0``, ..., ``C5`` as the "elements" of a ``3x2`` tile matrix.
The computation of ``C`` from ``A`` and ``B`` then follows the standard non-tiled matrix multiplication algorithm,
except that each "element" is itself a 2D tile, and each element-wise multiply is a smaller matrix multiplication.

This view fits neatly into the Tenstorrent architecture, where each Tensix core can perform matrix multiplication on two tiles in a single instruction.
All that needs to be done is to present the tiles of ``A`` and ``B`` to the matrix engine in the correct order, and accumulate results into the correct output tile.

Exercise 7: Implementing Matrix Multiplication in TT-Metalium
=============================================================

In this exercise, you will implement matrix multiplication on a Tenstorrent device. You can start with the lab_eltwise_binary example program and adjust it
to perform matrix multiplication.
Start by copying the files from the ``lab_eltwise_binary`` directory into a new directory (e.g. ``lab1_matmul``),
and rename the copied ``lab_eltwise_binary.cpp`` file to match the directory name (e.g. ``lab1_matmul.cpp``).
Similarly, rename ``tiles_add.cpp`` to e.g. ``tiles_matmul.cpp``.
Then, adjust the code to perform matrix multiplication, by making the following changes:

#. Update the host program to create input vectors to multiply matrix ``A`` of size ``640x320`` and matrix ``B``
   of size ``320x640`` to produce matrix ``C`` of size ``640x640``.

#. Copy the reference matrix multiplication code you created in Exercise 1.
   Adapt it to the ``bfloat16`` data type, so it can be used to verify TT-Metalium results.
   To limit precision loss, accumulate the result into an ordinary 32-bit ``float`` and cast
   to ``bfloat16`` only after the full sum is computed.

#. Update tensor creation code to create tensors of appropriate sizes for matrix multiplication and to
   pass required parameters to kernels (you may need to complete some of the other steps below to determine the correct parameters).
   You should write your code to make the following assumptions about the matrix and tile sizes:

   * Tiles will be square with dimensions ``TILE_HEIGHTxTILE_WIDTH`` (i.e. ``TILE_HEIGHT == TILE_WIDTH``).

   * All matrices will have dimensions that are divisible by the tile size.
     Note that constants ``TILE_HEIGHT`` and ``TILE_WIDTH`` are defined in the ``tt_metal/api/tt-metalium/constants.hpp`` header in the ``tt::constants`` namespace,
     and height is equal to width for all existing Tenstorrent devices.
     You should add assertions (using ``TT_FATAL``) that check these assumptions.

#. Update kernel creation code to refer to kernel ``.cpp`` files in the new directory.

#. Update the reader kernel to read the tiles of ``A`` and ``B`` in the correct order.
   The order of reading tiles from ``A`` and ``B`` should match the pattern of visiting one row of tiles of ``A``
   with all columns of tiles of ``B``, as discussed above. Keep in mind that ``noc_async_read_tile`` function only requires the index of the tile to read,
   not the actual memory address, so your code only needs to generate indices in the right order.

#. Update the writer kernel to write the tiles of ``C`` in the correct order. The order should match the pattern of visiting tiles of ``C`` in row-major order.
   Keep in mind that ``noc_async_write_tile`` function only requires the index of the tile to write, not the actual memory address,
   so your code only needs to generate indices in the right order.

#. Update the compute kernel to perform matrix multiplication rather than elementwise addition.
   To initialize the Tensix Engine for matrix multiplication, you will need to use the ``mm_init`` function provided in ``tt_metal/include/compute_kernel_api/matmul.h``.
   Do not use any other initialization functions for matrix multiplication (specifically do **not** use ``binary_op_init_common``, because that function is only
   applicable to elementwise operations, not to matrix multiplication).
   To multiply two tiles, you will need to use the ``matmul_tiles`` function provided in ``tt_metal/include/compute_kernel_api/matmul.h``.
   This function accumulates the result into the destination register; i.e. it adds to the existing values in the register rather than overwriting existing content.
   By judiciously choosing when to call ``tile_regs_acquire``, which initializes all tiles in the destination register array to zero, and when to call
   ``tile_regs_commit``, which signals that the compute core is done writing to the destination register,
   you can ensure that the result for each output tile is accumulated correctly.
   Don't forget to also pack each resulting tile and push it to the output circular buffer.
   Your compute kernel code should process the required number of tiles provided by reader kernels and
   produce the correct number of output tiles expected by the writer kernel.
   Remember that the JIT compiler can better optimize the kernel code if loop bounds are constant.
   Therefore, you should use compile-time arguments for the loop bounds whenever possible.

#. Update ``CMakeLists.txt`` in the new directory you created to specify the name of the new executable
   and the source files to compile, matching whatever file and directory names you chose.

#. Update ``CMakeLists.txt`` in the parent directory to add the new executable to be built.

#. Build your program by running ``./build_metal.sh`` from the ``tt-metal`` directory.

#. Run the program and verify the results by comparing the results with the reference matrix multiplication you created in Exercise 1.
   Note that because of the limited precision of bfloat16, the results may not be exactly the same as the reference results, but they should be
   numerically close, with relative differences on the order of a few percent for input data in the range of 0-1. Note that the relative difference
   may be higher if the reference solution doesn't use 32-bit ``float`` to accumulate the sum.

#. Profile the performance of the implementation, taking note of the elapsed firmware time. This will be useful to compare
   against future labs when we optimize the implementation for performance.
   If you previously used the ``--build-type Debug`` flag, **do not forget to rebuild the programming examples**
   with the ``--build-type Release`` flag, and also to disable DPRINT before profiling performance.


Conclusion
**********

This single-core matrix multiplication implementation highlights several key architectural patterns for programming Tenstorrent devices:

* **Separation of data movement and compute**: By using dedicated RISC-V processors for data movement (reader/writer kernels)
  and the matrix engine for computation, complex data orchestration patterns do not sacrifice compute throughput.
  The data movement processors can handle complex access patterns while the compute units remain fully utilized.
* **Tiled operations**: The hardware is optimized for tiled operations, making tile-based algorithms essential for achieving peak performance.
  All matrices are processed in tile units, matching the natural granularity of the underlying hardware accelerators.
* **Pipelined data movement**: The circular buffer architecture with double buffering enables overlapped execution - while the compute kernel
  processes current tiles, the data movement kernels can simultaneously fetch the next set of tiles.
  This pipelining ensures efficient utilization of compute resources by minimizing idle time.


Troubleshooting and Additional Resources
****************************************

In rare cases, a Tensix device may enter an undefined operational state if a program performs actions outside the supported behavior.
In such a case, the ``tt-smi -r`` command can be used to reset the device.
This operation restores the device to a clean state, allowing normal operation to resume.
If you encounter unexplained behaviors, try resetting the device using this command.
In an unlikely case that ``tt-smi -r`` gives an error, contact your system administrator.

Additional information about TT-Metalium and the Tenstorrent architecture can be found in the following resources:

* TT-Metalium Documentation: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html
* TT-Metalium GitHub Repository: https://github.com/tenstorrent/tt-metal
* TT-Metalium Discord: https://discord.gg/tenstorrent
