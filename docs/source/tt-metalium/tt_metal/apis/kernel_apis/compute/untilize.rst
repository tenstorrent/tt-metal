untilize
========

.. deprecated::
   The unpack-based ``untilize`` op is deprecated in favor of :doc:`pack_untilize <../pack_unpack/pack_untilize>`,
   which is significantly faster and in line with the programming model. It is scheduled for removal; see
   `tt-metal#22904 <https://github.com/tenstorrent/tt-metal/issues/22904>`_.

.. doxygenfunction:: untilize_init
.. doxygenfunction:: untilize_block
.. doxygenfunction:: untilize_uninit
