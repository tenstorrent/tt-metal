What is TT-NN?
#############

:doc:`TT-NN </ttnn/about>` is an open source library of neural network operations built using ``tt-metal`` programming model.

TT-NN can be used in C++ and Python projects and designed to be intuitive to developers familiar with `PyTorch <https://pytorch.org/>`_.

Key features of :doc:`TT-NN </ttnn/about>`:
    * More then 200 :doc:`operations </ttnn/api#operations>` including matrix multiplication, convolution, reduction, CCL, fused Transformer operations, etc
    * Tensor type that enables `different ways <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md>`_ to represent, distribute and access data on device
    * Developers can register custom operations
    * Native support for `mesh of devices <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md>`_
    * Tools to trace and `visualize <https://github.com/tenstorrent/ttnn-visualizer>`_ the computation `graph <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/graph-tracing.md>`_ of operations
    * Utilities to cache converted parameters to significantly speed up model loading on repeated runs.
    * `Comparison mode <https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/comparison-mode.md>`_ that allows to debug a long sequence of operations against a known reference
