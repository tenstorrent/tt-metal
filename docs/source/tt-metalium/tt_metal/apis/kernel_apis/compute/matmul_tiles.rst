matmul_tiles
============

.. doxygenfunction:: mm_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16, const uint32_t transpose=0)
.. doxygenfunction:: mm_init_short_with_dt(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t c_in_old_srca = 2, const uint32_t transpose=0)
.. doxygenfunction:: mm_init_short(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, const uint32_t transpose=0)
.. doxygenfunction:: matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst, const uint32_t transpose)
