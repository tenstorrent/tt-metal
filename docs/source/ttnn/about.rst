What is ttnn?
#############

:doc:`ttnn </ttnn/about>` is a library that provides a user-friendly interface to operations that run on TensTorrent's hardware using ``tt-metal`` programming model.

:doc:`ttnn </ttnn/about>` is designed to be intuitive to an user that is familiar with `PyTorch <https://pytorch.org/>`_.

:doc:`ttnn </ttnn/about>`'s primary dependency is :doc:`tt_lib </ttnn/dependencies/tt_lib>` which provides the implementation for all of the operations used by :doc:`ttnn </ttnn/about>`.

We trust that this library will be valuable to helping you on your journey to take full advantage of our devices!


What is the difference between :doc:`ttnn </ttnn/about>` and :doc:`tt_lib </ttnn/dependencies/tt_lib>`?
*******************************************************************************************************

:doc:`ttnn </ttnn/about>` is a wrapper around :doc:`tt_lib </ttnn/dependencies/tt_lib>`.

It utilizes :doc:`tt_lib </ttnn/dependencies/tt_lib>` data structures and operations and provides a user-friendly interface to them.

:doc:`tt_lib </ttnn/dependencies/tt_lib>` is meant to be fast-paced and to allow the user to be as flexible with the APIs as needed,
while :doc:`ttnn </ttnn/about>` is meant to be more stable and intuitive.

Here are key features that :doc:`ttnn </ttnn/about>` provides that aren't available in :doc:`tt_lib </ttnn/dependencies/tt_lib>`:
    * Support for N-D tensors.
    * More intuitive way of converting between :ref:`ttnn.ROW_MAJOR_LAYOUT<ttnn.ROW_MAJOR_LAYOUT>` and :ref:`ttnn.TILE_LAYOUT<ttnn.TILE_LAYOUT>` using :ref:`ttnn.to_layout<ttnn.to_layout>`
    * More constraints on the APIs to make them more intuitive without impacting the performance.
    * More stable APIs. Therefore, it is easier to write code that is compatible with future versions of the library.
    * The computation graph of :doc:`ttnn </ttnn/about>` operations can be traced and then visualized or used in any other way. The graph is `networkx` compatible. Refer to :ref:`ttnn Tracer<ttnn Tracer>` for examples
    * Infrastructure for preprocessing `torch.nn.Module` objects and extracting their parameters/modules. This infrastructure supports caching of the processed parameters which significantly speeds up repeated runs of the model.
    * Ability to compare the result of each operation to the equivalent `PyTorch <https://pytorch.org/>`_ operation. Very useful for debugging.


In case :doc:`ttnn </ttnn/about>` doesn't provide an operation that is needed, then the tensor can be easily converted to :doc:`tt_lib </ttnn/dependencies/tt_lib>` tensor:

.. code-block:: python

    tensor: ttnn.Tensor                              # ttnn.Tensor from some operation
    ttl_tensor: tt_lib.tensor.Tensor = tensor.value  # Convert to tt_lib.tensor.Tensor
    tensor = ttnn.Tensor(tensor)                     # Convert back to ttnn.Tensor
