.. _Getting Started:

Getting Started
===============

TT-Metalium is a framework for accelerating both ML and non-ML workloads on Tenstorrent hardware.

TT-Metalium provides a C++ API for developers to write custom kernels that runs on Tenstorrent hardware. It gives you full control over the hardware, allowing you to optimize your code for performance and efficiency. And it does not hide the hardware details from you, so you can write code that is tailored to your specific use case.

The GitHub repository for the project is located here:
https://github.com/tenstorrent/tt-metal

Installation
------------

Install dependencies and build the project by following the instructions in the `installation guide
<../installing.html>`_.


Quick Start Guide
-----------------

Basic Usage
^^^^^^^^^^^

**Step 1: DRAM Loopback**
    Learn the basic structure of an Metalium application by implementing a :ref:`DRAM Loopback Example` that copies data from one DRAM buffer to another (hence loopback). This example will help you understand the basic concepts fundamental to writing Metalium applications.

    **What you'll learn:** Basic host and kernel structure, buffer management, and data transfer.

**Step 2: Eltwise Binary Kernel**
    Build on the loopback example by implementing an :ref:`Eltwise Binary Kernel<Eltwise binary example>` that performs element-wise addition of two buffers. This will introduce you to performing computations using the matrix engine (FPU) and passing data between kernels within a single Tensix core.

    **What you'll learn:** Circular buffer for data passing, compute kernels, using the matrix engine for computations.

**Step 3: Eltwise SFPU**
    Extend the previous example to implement an :ref:`Eltwise SFPU<Eltwise sfpu example>` kernel that performs element-wise addition using the SFPU (vector engine, Special Function Processing Unit). This will introduce you to the SFPU and how to use it for vectorized operations.

    **What you'll learn:** Performing operations using the SFPU.

Intermediate Usage
^^^^^^^^^^^^^^^^^^
**Step 4: Single-core Matrix Multiplication**
    Implement a :ref:`Single-core Matrix Multiplication Kernel<MatMul_Single_Core example>` that performs matrix multiplication using the matrix engine. This will help you understand how to handle complex dataflow and computations on the Tensix core.

    **What you'll learn:** Complex dataflow, tilized operatins and using the matrix engine for matrix multiplication.

Advanced Usage
^^^^^^^^^^^^^^

**Step 5: Multi-core Matrix Multiplication**
    Build on the single-core matrix multiplication example to implement a :ref:`Multi-core Matrix Multiplication Kernel<MatMul_Multi_Core example>` that distributes the workload across multiple Tensix cores. This will introduce you to parallel processing and how to optimize performance by leveraging multiple cores.

    **What you'll learn:** Parallel processing and splitting workloads across multiple cores.

**Step 6: Optimized Multi-core Matrix Multiplication**
    :ref:`Optimize the multi-core matrix multiplication<MatMul_Multi_Core_example>` kernel by implementing exploting the grid structure of the processor. Avoid redundant reads from DRAM and avoid NoC congestion by reading with one core and broadcasting the data to other cores. This will help you understand how to optimize performance by minimizing unneeded data movement and maximizing data reuse.

    **What you'll learn:** Performance optimization techniques for high-performance kernels.

Next Steps
----------

**For ML Developers**
    Use the higher-level `TT-NN <../../ttnn>`_ API for model development and deployment.

**For Contributors**
    Review the `contribution guidelines <https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md>`_ before submitting changes. We are always happy to assist in merging your contributions.

**For Custom Kernels**
    Study the provided examples and adapt them to your specific use case. Start simple and optimize iteratively.
