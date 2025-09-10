.. _Tutorials:

Tutorials
#########

This section provides a collection of Python tutorials designed to help you get started with **TT-NN** for tasks such as tensor operations, model conversion, and inference.

To run these tutorials smoothly, we recommend using a Python virtual environment with the necessary dependencies installed. You can set this up in one of two ways:

- **Full Development Environment:**
  Follow the instructions in the
  `TT-NN / TT-Metal Installation Guide <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>`_
  to set up the complete development environment.

- **Lightweight Tutorial Environment:**
  Use the provided
  `Python Environment Setup Script <https://github.com/tenstorrent/tt-metal/tree/main/ttnn/tutorials/basic_python/tutorials_venv.sh>`_
  to create a minimal virtual environment specifically for running the tutorials.

Each tutorial also has an equivalent standalone Python script that you can run locally. These scripts are located in the
`ttnn/tutorials/basic_python/ <https://github.com/tenstorrent/tt-metal/tree/main/ttnn/tutorials/basic_python>`_
directory of the **TT-Metal** repository.

With your virtual environment activated, you can run the tutorials directly:

.. code-block:: console

   $ python3 --version
   Python 3.10.12
   $ python3 example.py
   ...

Available tutorials:

.. toctree::

   tutorials/2025_dx_rework/ttnn_add_tensors.ipynb
   tutorials/2025_dx_rework/ttnn_basic_operations.ipynb
   tutorials/2025_dx_rework/ttnn_mlp_inference_mnist.ipynb
   tutorials/2025_dx_rework/ttnn_multihead_attention.ipynb
   tutorials/2025_dx_rework/ttnn_basic_conv.ipynb
   tutorials/2025_dx_rework/ttnn_simplecnn_inference.ipynb
   tutorials/2025_dx_rework/ttnn_visualizer.md
