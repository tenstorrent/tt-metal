matmul_block
============

.. doxygenfunction:: matmul_block_init
.. doxygenfunction:: matmul_block(std::uint32_t in0_cb_id, std::uint32_t in1_cb_id, std::uint32_t in0_tile_index, std::uint32_t in1_tile_index, std::uint32_t idst, const std::uint32_t transpose, std::uint32_t ct_dim, std::uint32_t rt_dim, std::uint32_t kt_dim, std::uint32_t call_line = __builtin_LINE())

See also :doc:`compute_kernel_hw_startup`, which must be called once with ``SrcOrder::Reverse`` before ``matmul_block_init``.
