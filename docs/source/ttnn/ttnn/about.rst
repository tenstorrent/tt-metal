What is ttnn?
#############

:doc:`ttnn </ttnn/about>` is a library that provides a user-friendly interface to operations that run on TensTorrent's hardware using ``tt-metal`` programming model.

:doc:`ttnn </ttnn/about>` is designed to be intuitive to an user that is familiar with `PyTorch <https://pytorch.org/>`_.

:doc:`ttnn </ttnn/about>`'s primary dependency is :doc:`tt_lib </ttnn/dependencies/tt_lib>` which provides the implementation for all of the operations used by :doc:`ttnn </ttnn/about>`.

We trust that this library will be valuable to helping you on your journey to take full advantage of our devices!


Key features of :doc:`ttnn </ttnn/about>`
*****************************************

Key features of :doc:`ttnn </ttnn/about>`:
    * Support for N-D tensors.
    * Intuitive way of converting between :ref:`ttnn.ROW_MAJOR_LAYOUT<ttnn.ROW_MAJOR_LAYOUT>` and :ref:`ttnn.TILE_LAYOUT<ttnn.TILE_LAYOUT>` using :ref:`ttnn.to_layout<ttnn.to_layout>`
    * Stable APIs.
    * The computation graph of :doc:`ttnn </ttnn/about>` operations can be traced and then visualized or used in any other way. The graph is `networkx` compatible. Refer to :ref:`ttnn Tracer<ttnn Tracer>` for examples
    * Infrastructure for converting parameters and some sub-modules from a `torch.nn.Module` object. This infrastructure supports caching of the converted parameters which could significantly speed up repeated runs.
    * Ability to compare the result of each operation to the equivalent `PyTorch <https://pytorch.org/>`_ operation. Very useful for debugging.
