matmul_tiles
============

.. doxygenfunction:: matmul_init
.. doxygenfunction:: matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst)

See also :doc:`compute_kernel_hw_startup`, which must be called once with ``SrcOrder::Reverse`` before ``matmul_init``.
