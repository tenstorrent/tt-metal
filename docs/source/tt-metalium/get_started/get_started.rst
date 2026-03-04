.. _Getting Started:

Getting Started
===============

TT-Metalium is a framework for accelerating both ML and non-ML workloads on Tenstorrent hardware.
It offers an abstraction between the host system (e.g., desktop with x86 CPU) and Tenstorrent hardware.

TT-Metalium provides a C++ API for writing kernels that run on Tensix hardware; for configuring these kernels; and for executing kernels on the hardware. It gives the programmer full control over the hardware and data movement. This allows code to be optimized for performance and efficiency. Since hardware details are exposed, the programmer can write code that is tailored to their specific use case.

The GitHub repository for the project is located here:
https://github.com/tenstorrent/tt-metal

Software Stack Overview
-----------------------

TT-Metalium sits at the foundation of Tenstorrent's software stack:

- **TT-Forge / TT-MLIR**: High-level compilation frameworks for deploying neural networks (see https://github.com/tenstorrent/tt-forge-fe)
- **TTNN**: Library of kernels implementing common Machine Learning operations (see https://docs.tenstorrent.com/tt-metal/latest/ttnn/)
- **TT-Metalium**: Low-level programming interface for Tensix hardware ⬅ This guide
- **TT-LLK (Low Level Kernels)**: Hardware-specific kernel implementations

Programming Philosophy
----------------------

Operations on Tenstorrent hardware are typically designed from the **bottom up**:

1. Start with a kernel on a single Tensix core
2. Schedule the kernel across multiple Tensix cores (may require synchronization)
3. Scale to multiple devices (critical for large model deployment)

The examples in this guide follow this progression, starting simple and building complexity.

Installation
------------

Install dependencies and build the project by following the instructions in the `installation guide
<../installing.html>`_.

Key Concepts
------------

**Pipeline Dataflow Pattern**

Kernels on Tenstorrent hardware typically follow a three-stage pipeline:

1. **Reader Kernel** (Data Movement): Reads data from DRAM/SRAM into circular buffers
2. **Compute Kernel**: Processes data using Matrix/Vector engines
3. **Writer Kernel** (Data Movement): Writes results back to DRAM/SRAM

**Circular buffers** enable communication and synchronization between kernels, acting as FIFO data structures. They allow the reader to fetch new data while compute processes previous data, enabling overlapped execution.

The DRAM Loopback example (Step 1 below) demonstrates the basic data movement pattern. Step 2 introduces the full pipeline with compute kernels and circular buffers.

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

Resources and Debugging
-----------------------

Documentation
^^^^^^^^^^^^^

- **Metalium Architecture Guide**: Comprehensive guide to architecture and programming model: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- **API Reference**: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/

Related Repositories
^^^^^^^^^^^^^^^^^^^^

- **tt-metal**: Main repository with examples and documentation - https://github.com/tenstorrent/tt-metal
- **tt-forge**: High-level neural network compiler framework - https://github.com/tenstorrent/tt-forge-fe
- **tt-isa-documentation**: Hardware instruction set architecture documentation - https://github.com/tenstorrent/tt-isa-documentation/

Debugging Tools
^^^^^^^^^^^^^^^

For information on debugging and profiling tools (DPRINT, Tracy Profiler, Device Profiler, Inspector, Watcher, etc.), see the :doc:`Tools documentation <../tools/index>`.

Community Support
^^^^^^^^^^^^^^^^^

- **Documentation**: https://docs.tenstorrent.com/
- **GitHub Issues**: https://github.com/tenstorrent/tt-metal/issues

Next Steps
----------

**For ML Developers**
    Use the higher-level `TT-NN <https://docs.tenstorrent.com/tt-metal/latest/ttnn/>`_ API for model development and deployment. TT-NN builds on TT-Metalium but provides PyTorch-like operations and automatic kernel selection.

**For Architecture Deep Dives**
    Read the comprehensive `Metalium Architecture Guide <https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md>`_ for deep dives into hardware architecture, NoC topology, and advanced programming patterns.

**For Contributors**
    Review the `contribution guidelines <https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md>`_ before submitting changes. We are always happy to assist in merging your contributions.

**For Custom Kernels**
    Study the provided examples and adapt them to your specific use case. Start simple and optimize iteratively.
