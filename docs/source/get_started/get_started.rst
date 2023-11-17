.. _Getting Started:

Getting Started
===============

The GitHub page for the project is located here:
https://github.com/tenstorrent-metal/tt-metal

To be added to the repo, please contact Jasmina or Davor at:
jvasiljevic@tenstorrent.com, dcapalija@tenstorrent.com.

Installation and environment setup instructions are in the GitHub repository
front-page README: https://github.com/tenstorrent-metal/tt-metal#installing

Welcome to tt-metal Quick Start Guide
=====================================

Metal provides developers to do more than running models, facilitating a transition from running models effortlessly out of the box, engaging in lightweight optimizations, and progressing into more sophisticated, heavyweight optimizations. This series of five steps serves as an illustrative example, showcasing the available tools for optimizing performance on Tenstorrent hardware

1. **Install and Build:**
   Install tt-metal and build the project by following the instructions in the `installation guide <https://github.com/tenstorrent-metal/tt-metal#installing>`_.

2. **Explore the Falcon 7B Demo:**
   Get started with the Falcon 7B demo to experience the capabilities of tt-metal. Navigate to the `Falcon 7B demo folder <https://github.com/tenstorrent-metal/tt-metal/tree/main/models/demos/falcon7b>`_ for details.

3. **ttnn Tutorial: Multi-Head Attention (Simple):**
   Learn the basics of multi-head attention operations in tt-metal's ttnn module with a simple example. Follow the tutorial `here <https://tenstorrent-metal.github.io/tt-metal/latest/ttnn/tutorials/multihead-attention.html#multi-head-attention>`_.

4. **ttn Tutorial: Multi-Head Attention (Optimized):**
   Dive deeper into multi-head attention operations in ttn, optimizing performance. Coming soon.

5. **Advanced Metal Usage: Matrix Multiplication Kernels:**
   Explore expert-level usage by working with Metal to define your own matrix multiplication kernels. Choose between `single-core <https://github.com/tenstorrent-metal/tt-metal/blob/main/docs/source/tt_metal/examples/matmul_single_core.rst>`_ and `multi-core <https://github.com/tenstorrent-metal/tt-metal/blob/main/docs/source/tt_metal/examples/matmul_multi_core.rst>`_ implementations.

If you're an ML developer and looking for further docs for using the Python
library APIs to build models, please now go to :ref:`TT-Metal Models Get
Started`.

If you're an internal TT-Metal developer, please now read please review the
`contribution standards
<https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md>`_.
