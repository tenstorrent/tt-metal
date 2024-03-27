.. _Getting Started:

Getting Started
===============

TT-Metalium is designed with the needs for non-ML and ML use cases.

The GitHub page for the project is located here:
https://github.com/tenstorrent-metal/tt-metal

Installation and environment setup instructions are in the GitHub repository
front-page README: https://github.com/tenstorrent-metal/tt-metal#installing

Quick Start Guide
-----------------

Metalium provides developers to do more than running models, facilitating a
transition from running models effortlessly out of the box, engaging in
lightweight optimizations, and progressing into more sophisticated, heavyweight
optimizations. This series of five steps serves as an illustrative example,
showcasing the available tools for optimizing performance on Tenstorrent
hardware.

1. Install and Build
^^^^^^^^^^^^^^^^^^^^

Install tt-metal and build the project by following the instructions in the
`installation guide
<https://github.com/tenstorrent-metal/tt-metal#installing>`_.

2. Explore the Falcon 7B Demo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get started with the Falcon 7B demo to experience the capabilities of tt-metal.
Navigate to the `Falcon 7B demo folder
<https://github.com/tenstorrent-metal/tt-metal/tree/main/models/demos/falcon7b>`_
for details.

You can also check our demos for
`ResNet <https://github.com/tenstorrent-metal/tt-metal/tree/main/models/demos/resnet>`_,
`BERT <https://github.com/tenstorrent-metal/tt-metal/tree/main/models/demos/metal_BERT_large_11>`_,
`Mistral 7B <https://github.com/tenstorrent-metal/tt-metal/tree/main/models/demos/mistral7b>`_,
and
`Llama2-70B <https://github.com/tenstorrent-metal/tt-metal/tree/main/models/demos/llama2_70b>`_.

3. ttnn Tutorial: Multi-Head Attention (Simple)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Learn the basics of multi-head attention operations in tt-metal's ttnn module
with a simple example: `ttnn simple module <../../ttnn/ttnn/tutorials/ttnn_tutorials/003.html#Write-Multi-Head-Attention-using-ttnn>`_.

4. ttnn Tutorial: Multi-Head Attention (Optimized)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dive deeper into multi-head attention operations in ttnn, optimizing
performance: `optimizing performance <../../ttnn/ttnn/tutorials/ttnn_tutorials/003.html#Write-optimized-version-of-Multi-Head-Attention>`_.

5. Advanced Metalium Usage: Matrix Multiplication Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Explore expert-level usage by working with Metalium to define your own matrix
multiplication kernels. Choose between :ref:`single-core
<MatMul_Single_Core example>`
and :ref:`multi-core<MatMul_Multi_Core example>`
implementations.

Where to go from here
---------------------

If you're an ML developer and looking for further docs for using the Python
library APIs to build models, please now go to `getting started for models <../../ttnn/tt_metal_models/get_started.html>`_.

If you're an internal TT-Metalium developer, please now read please review the
`contribution standards
<https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md>`_.
