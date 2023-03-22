# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_TESTS += \
		 test/tt_metal/ops/test_eltwise_binary_op \
		 test/tt_metal/ops/test_eltwise_unary_op \
		 test/tt_metal/dtx/unit_tests \
		 test/tt_metal/dtx/tensor \
		 test/tt_metal/dtx/overlap \
		 test/tt_metal/dtx/collapse_transformations \
		 test/tt_metal/ops/test_transpose_op \
		 test/tt_metal/ops/test_transpose_wh_single_core \
		 test/tt_metal/ops/test_transpose_wh_multi_core \
		 test/tt_metal/ops/test_transpose_hc_rm_8bank_single_core \
		 test/tt_metal/ops/test_transpose_hc_rm_8bank_multi_core \
		 test/tt_metal/ops/test_reduce_op \
		 test/tt_metal/ops/test_bcast_op \
		 test/tt_metal/ops/test_bmm_op \
		 test/tt_metal/ops/test_tilize_op \
		 test/tt_metal/ops/test_tilize_zero_padding \
		 test/tt_metal/ops/test_tilize_op_channels_last \
		 test/tt_metal/ops/test_tilize_zero_padding_channels_last \
		 test/tt_metal/ops/test_tilize_conv_activation \
		 test/tt_metal/test_bmm \
		 test/tt_metal/tensors/test_host_device_loopback \
		 test/tt_metal/test_add_two_ints \
		 test/tt_metal/test_dram_to_l1_multicast \
		 test/tt_metal/test_dram_to_l1_multicast_loopback_src \
		 test/tt_metal/test_dram_loopback_single_core \
		 test/tt_metal/test_dram_loopback_single_core_db \
		 test/tt_metal/test_eltwise_binary \
		 test/tt_metal/test_matmul_single_tile \
		 test/tt_metal/test_matmul_multi_tile \
		 test/tt_metal/test_matmul_large_block \
		 test/tt_metal/test_matmul_single_core \
		 test/tt_metal/test_matmul_multi_core_single_dram \
		 test/tt_metal/test_matmul_multi_core_multi_dram \
		 test/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast \
		 test/tt_metal/test_matmul_multi_core_multi_dram_in1_mcast \
		 test/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast \
		 test/tt_metal/test_datacopy \
		 test/tt_metal/test_dataflow_cb \
		 test/tt_metal/test_flatten \
		 test/tt_metal/test_transpose_hc \
		 test/tt_metal/test_transpose_wh \
		 test/tt_metal/test_multiple_programs \
		 test/tt_metal/test_multi_core_kernel \
		 test/tt_metal/test_unpack_tilize \
		 test/tt_metal/test_unpack_untilize \
		 test/tt_metal/test_graph_interpreter \
		 test/tt_metal/test_interleaved_layouts \
		 test/tt_metal/test_bcast \
		 test/tt_metal/test_dtx \
		 test/tt_metal/test_dtx_tilized_row_to_col_major \
		 test/tt_metal/test_sfpu \
		 test/tt_metal/test_generic_binary_reader_matmul_large_block \
		 test/tt_metal/test_3x3conv_as_matmul_large_block \
		 test/tt_metal/test_pipeline_across_rows \
		 test/tt_metal/test_l1_to_l1_multi_core \
		 test/tt_metal/test_dram_copy_sticks_multi_core \
		 test/tt_metal/test_reduce_h \
		 test/tt_metal/test_reduce_w \
		 test/tt_metal/test_reduce_hw \
		 test/tt_metal/test_untilize_eltwise_binary \
		 # test/tt_metal/test_datacopy_multi_core_multi_dram \  # this does not compile

TT_METAL_TESTS_SRCS = $(addprefix test/tt_metal/, $(addsuffix .cpp, $(TT_METAL_TESTS:test/%=%)))

TT_METAL_TEST_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES) -Itt_gdb -Itt_metal/tests -Icompile_trisc -Iverif
TT_METAL_TESTS_LDFLAGS = -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -lhlkc_api -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

TT_METAL_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.o))
TT_METAL_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
test/tt_metal: $(TT_METAL_TESTS) programming_examples
test/tt_metal/all: $(TT_METAL_TESTS)
test/tt_metal/%: $(TESTDIR)/tt_metal/% ;

.PRECIOUS: $(TESTDIR)/tt_metal/%
$(TESTDIR)/tt_metal/%: $(OBJDIR)/tt_metal/test/%.o $(BACKEND_LIB) $(TT_METAL_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_metal/test/%.o
$(OBJDIR)/tt_metal/test/%.o: test/tt_metal/tt_metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -c -o $@ $<

tt_metal/tests: test/tt_metal
tt_metal/tests/all: test/tt_metal/all
