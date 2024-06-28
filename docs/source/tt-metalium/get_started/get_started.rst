.. _Getting Started:

Getting Started
===============

TT-Metalium is designed with the needs for non-ML and ML use cases.

The GitHub page for the project is located here:
https://github.com/tenstorrent/tt-metal

Installation and environment setup instructions are in the GitHub repository
front-page README: https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md

Quick Start Guide
-----------------

Metalium provides developers to do more than running models, facilitating a
transition from running models effortlessly out of the box, engaging in
lightweight optimizations, and progressing into more sophisticated, heavyweight
optimizations. This series of steps serves as an illustrative example,
showcasing the available tools for optimizing performance on Tenstorrent
hardware.

1. Install and Build
^^^^^^^^^^^^^^^^^^^^

Install and build the project by following the instructions in the
`installation guide
<../installing.html>`_.

2. Beginner Metalium Usage: DRAM Loopback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Try creating a :ref:`basic kernel example <DRAM Loopback Example>` that uses
the L1 and DRAM memory structures of the Tenstorrent device.

3. Beginner Metalium Usage: Eltwise Binary Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Augment your loopback example an :ref:`additional kernel <Eltwise binary
example>` that will use the compute engine of the Tensix core to add values in
two buffers.

4. Beginner Metalium Usage: Single-core Matrix Multiplication Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use TT-Metalium to define your own matrix multiplication kernels. Refer to our
simpler :ref:`single-core <MatMul_Single_Core example>` example as a starting
point.

5. Advanced Metalium Usage: Multi-core Matrix Multiplication Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Explore expert-level usage by building on the previous example to create a
:ref:`multi-core<MatMul_Multi_Core example>` implementation.

Where to go from here
^^^^^^^^^^^^^^^^^^^^^

If you're an ML developer and looking for a simpler Python API to build models,
take a look at our higher-level API `TT-NN <../../ttnn>`_.

If you're an internal TT-Metalium developer, please now read and review the
`contribution standards
<https://github.com/tenstorrent/tt-metal/blob/main/CONTRIBUTING.md>`_.
