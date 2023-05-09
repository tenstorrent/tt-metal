# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_TESTS += \
		 tests/tt_metal/ops/test_eltwise_binary_op \
		 tests/tt_metal/ops/test_eltwise_unary_op \
		 tests/tt_metal/dtx/unit_tests \
		 tests/tt_metal/dtx/tensor \
		 tests/tt_metal/dtx/overlap \
		 tests/tt_metal/dtx/collapse_transformations \
		 tests/tt_metal/ops/test_softmax_op \
		 tests/tt_metal/ops/test_layernorm_op \
		 tests/tt_metal/ops/test_transpose_op \
		 tests/tt_metal/ops/test_transpose_wh_single_core \
		 tests/tt_metal/ops/test_transpose_wh_multi_core \
		 tests/tt_metal/ops/test_transpose_hc_rm_8bank_single_core \
		 tests/tt_metal/ops/test_transpose_hc_rm_8bank_multi_core \
		 tests/tt_metal/ops/test_reduce_op \
		 tests/tt_metal/ops/test_bcast_op \
		 tests/tt_metal/ops/test_bmm_op \
		 tests/tt_metal/ops/test_tilize_op \
		 tests/tt_metal/ops/test_tilize_zero_padding \
		 tests/tt_metal/ops/test_tilize_op_channels_last \
		 tests/tt_metal/ops/test_tilize_zero_padding_channels_last \
		 tests/tt_metal/test_bmm \
		 tests/tt_metal/tensors/test_copy_and_move \
		 tests/tt_metal/tensors/test_host_device_loopback \
		 tests/tt_metal/allocator/test_free_list_allocator_algo \
		 tests/tt_metal/allocator/test_l1_banking_allocator \
		 tests/tt_metal/test_add_two_ints \
		 tests/tt_metal/test_dram_to_l1_multicast \
		 tests/tt_metal/test_dram_to_l1_multicast_loopback_src \
		 tests/tt_metal/test_dram_loopback_single_core \
		 tests/tt_metal/test_dram_loopback_single_core_db \
		 tests/tt_metal/test_eltwise_binary \
		 tests/tt_metal/test_matmul_single_tile_bfp8b \
		 tests/tt_metal/test_matmul_single_tile \
		 tests/tt_metal/test_matmul_single_tile_output_in_l1 \
		 tests/tt_metal/test_matmul_multi_tile \
		 tests/tt_metal/test_matmul_large_block \
		 tests/tt_metal/test_matmul_single_core \
		 tests/tt_metal/test_matmul_multi_core_single_dram \
		 tests/tt_metal/test_matmul_multi_core_multi_dram \
		 tests/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast \
		 tests/tt_metal/test_matmul_multi_core_multi_dram_in1_mcast \
		 tests/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast \
		 tests/tt_metal/test_datacopy_bfp8b \
		 tests/tt_metal/test_datacopy \
		 tests/tt_metal/test_datacopy_output_in_l1 \
		 tests/tt_metal/test_dataflow_cb \
		 tests/tt_metal/test_flatten \
		 tests/tt_metal/test_transpose_hc \
		 tests/tt_metal/test_transpose_wh \
		 tests/tt_metal/test_multiple_programs \
		 tests/tt_metal/test_multi_core_kernel \
		 tests/tt_metal/test_unpack_tilize \
		 tests/tt_metal/test_unpack_untilize \
		 tests/tt_metal/test_graph_interpreter \
		 tests/tt_metal/test_interleaved_layouts \
		 tests/tt_metal/test_interleaved_l1_buffer \
		 tests/tt_metal/test_bcast \
		 tests/tt_metal/test_dtx \
		 tests/tt_metal/test_dtx_tilized_row_to_col_major \
		 tests/tt_metal/test_sfpu \
		 tests/tt_metal/test_generic_binary_reader_matmul_large_block \
		 tests/tt_metal/test_3x3conv_as_matmul_large_block \
		 tests/tt_metal/test_pipeline_across_rows \
		 tests/tt_metal/test_l1_to_l1_multi_core \
		 tests/tt_metal/test_dram_copy_sticks_multi_core \
		 tests/tt_metal/test_reduce_h \
		 tests/tt_metal/test_reduce_w \
		 tests/tt_metal/test_reduce_hw \
		 tests/tt_metal/test_untilize_eltwise_binary \
		 tests/tt_metal/test_bfp8_conversion \
		 tests/tt_metal/test_semaphores \
		 tests/tt_metal/tt_dispatch/test_enqueue_read_and_write \
		 # test/tt_metal/test_datacopy_multi_core_multi_dram \  # this does not compile

TT_METAL_TESTS_SRCS = $(addprefix tests/tt_metal/, $(addsuffix .cpp, $(TT_METAL_TESTS:tests/%=%)))

TT_METAL_TEST_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES)
TT_METAL_TESTS_LDFLAGS = -ltensor -ltt_dnn -ldtx -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp -ltt_dispatch

TT_METAL_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.o))
TT_METAL_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/tt_metal: $(TT_METAL_TESTS) programming_examples
tests/tt_metal/all: $(TT_METAL_TESTS)
tests/tt_metal/%: $(TESTDIR)/tt_metal/% ;

.PRECIOUS: $(TESTDIR)/tt_metal/%
$(TESTDIR)/tt_metal/%: $(OBJDIR)/tt_metal/tests/%.o $(TT_METAL_LIB) $(TT_DNN_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_metal/tests/%.o
$(OBJDIR)/tt_metal/tests/%.o: tests/tt_metal/tt_metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -c -o $@ $<

tt_metal/tests: tests/tt_metal
tt_metal/tests/all: tests/tt_metal/all
