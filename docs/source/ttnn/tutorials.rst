Tutorials
#########

This is a collection of tutorials written with Jupyter Notebooks to help you ramp up your skillset for using `tt-metal`. These
notebooks can be found under https://github.com/tenstorrent-metal/tt-metal/tree/main/ttnn/tutorials.

These tutorials assume you already have a machine set up with either a grayskull or wormhole device available and that you have successfully
followed the instructions for `installing and building the software <https://github.com/tenstorrent-metal/tt-metal/blob/main/README.md>`_.

If you would like to see logs showing the time it takes for the device to execute a program, please run `export TTNN_ENABLE_LOGGING=1` and re-compile after running `make clean`

From within the `ttnn/tutorials` directory, launch the notebooks with: :code:`jupyter lab --no-browser --port=8888`
Hint: Be sure to always run the cells from top to bottom as the order of the cells are dependent.

.. toctree::

   tutorials/add.rst
   tutorials/matmul.rst
   tutorials/multihead-attention.rst
