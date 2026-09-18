matmul_block
============

.. doxygenfunction:: matmul_block_init
.. doxygenfunction:: matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim, uint32_t call_line = __builtin_LINE())

See also :doc:`compute_kernel_hw_startup`, which must be called once with ``SrcOrder::Reverse`` before ``matmul_block_init``.
