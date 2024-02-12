matmul_tiles
============

.. doxygenfunction:: mm_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16, const uint32_t transpose=0)
.. doxygenfunction:: mm_init_short_with_dt(uint32_t c_in0 = 0, uint32_t c_in1 = 1, uint32_t c_in_old_srca = 2, const uint32_t transpose=0)
.. doxygenfunction:: mm_init_short(uint32_t c_in0 = 0, uint32_t c_in1 = 1, const std::uint32_t transpose=0)
.. doxygenfunction:: matmul_tiles(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose)
