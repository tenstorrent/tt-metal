.. _Tutorials:

.. note::
   Certain TT-NN tutorials currently work on Grayskull only. Please check the specific pages
   of tutorials below for more information.

Tutorials
#########

This is a collection of tutorials written with Jupyter Notebooks to help you ramp up your skillset for using `tt-metal`. These
notebooks can be found under https://github.com/tenstorrent/tt-metal/tree/main/ttnn/tutorials.

These tutorials assume you already have a machine set up with either a grayskull or wormhole device available and that you have successfully
followed the instructions for `installing and building the software from source <https://github.com/tenstorrent/tt-metal/blob/main/README.md>`_.

From within the `ttnn/tutorials` directory, launch the notebooks with: :code:`jupyter lab --no-browser --port=8888`
Hint: Be sure to always run the cells from top to bottom as the order of the cells are dependent.

.. toctree::

   tutorials/tensor_and_add_operation.rst
   tutorials/matmul.rst
   tutorials/multihead-attention.rst
   tutorials/ttnn-tracer.rst
   tutorials/profiling.rst
   tutorials/resnet-basic-block.rst
   tutorials/graphing_torch_dit.rst
