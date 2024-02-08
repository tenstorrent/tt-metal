matmul_block
============

.. doxygenfunction:: mm_block_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)
.. doxygenfunction:: mm_block_init_short(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, const std::uint32_t transpose=0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)
.. doxygenfunction:: mm_block_init_short_with_dt(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t cbid=2, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)
.. doxygenfunction:: matmul_block(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim)
