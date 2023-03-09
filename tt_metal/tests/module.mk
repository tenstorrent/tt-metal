# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_TESTS += \
		 tt_metal/tests/ops/test_eltwise_binary_op \
		 tt_metal/tests/ops/test_eltwise_unary_op \
		 tt_metal/tests/dtx/unit_tests \
		 tt_metal/tests/dtx/tensor \
		 tt_metal/tests/dtx/overlap \
		 tt_metal/tests/dtx/collapse_transformations \
		 tt_metal/tests/ops/test_transpose_op \
		 tt_metal/tests/ops/test_transpose_wh_single_core \
		 tt_metal/tests/ops/test_transpose_wh_multi_core \
		 tt_metal/tests/ops/test_transpose_hc_rm_8bank_single_core \
		 tt_metal/tests/ops/test_transpose_hc_rm_8bank_multi_core \
		 tt_metal/tests/ops/test_reduce_op \
		 tt_metal/tests/ops/test_bcast_op \
		 tt_metal/tests/ops/test_bmm_op \
		 tt_metal/tests/ops/test_tilize_op \
		 tt_metal/tests/ops/test_tilize_zero_padding \
		 tt_metal/tests/test_bmm \
		 tt_metal/tests/tensors/test_host_device_loopback \
		 tt_metal/tests/test_add_two_ints \
		 tt_metal/tests/test_dram_to_l1_multicast \
		 tt_metal/tests/test_dram_to_l1_multicast_loopback_src \
		 tt_metal/tests/test_dram_loopback_single_core \
		 tt_metal/tests/test_dram_loopback_single_core_db \
		 tt_metal/tests/test_eltwise_binary \
		 tt_metal/tests/test_matmul_single_tile \
		 tt_metal/tests/test_matmul_multi_tile \
		 tt_metal/tests/test_matmul_large_block \
		 tt_metal/tests/test_matmul_single_core \
		 tt_metal/tests/test_matmul_multi_core_single_dram \
		 tt_metal/tests/test_matmul_multi_core_multi_dram \
		 tt_metal/tests/test_matmul_multi_core_multi_dram_in0_mcast \
		 tt_metal/tests/test_matmul_multi_core_multi_dram_in1_mcast \
		 tt_metal/tests/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast \
		 tt_metal/tests/test_datacopy \
		 tt_metal/tests/test_dataflow_cb \
		 tt_metal/tests/test_flatten \
		 tt_metal/tests/test_transpose_hc \
		 tt_metal/tests/test_transpose_wh \
		 tt_metal/tests/test_multiple_programs \
		 tt_metal/tests/test_multi_core_kernel \
		 tt_metal/tests/test_unpack_tilize \
		 tt_metal/tests/test_unpack_untilize \
		 tt_metal/tests/test_graph_interpreter \
		 tt_metal/tests/test_interleaved_layouts \
		 tt_metal/tests/test_bcast \
		 tt_metal/tests/test_dtx \
		 tt_metal/tests/test_dtx_tilized_row_to_col_major \
		 tt_metal/tests/test_sfpu \
		 tt_metal/tests/test_generic_binary_reader_matmul_large_block \
		 tt_metal/tests/test_3x3conv_as_matmul_large_block \
		 tt_metal/tests/test_pipeline_across_rows \
		 tt_metal/tests/test_l1_to_l1_multi_core \
		 tt_metal/tests/test_dram_copy_sticks_multi_core \
		 tt_metal/tests/test_reduce_h \
		 tt_metal/tests/test_reduce_w \
		 tt_metal/tests/test_reduce_hw \

TT_METAL_TESTS_SRCS = $(addsuffix .cpp, $(TT_METAL_TESTS))

TT_METAL_TEST_INCLUDES = $(TT_METAL_INCLUDES) -Itt_gdb -Itt_metal/tests -Icompile_trisc -Iverif
TT_METAL_TESTS_LDFLAGS = -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -lhlkc_api -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

TT_METAL_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.o))
TT_METAL_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal/tests: $(TT_METAL_TESTS) programming_examples
tt_metal/tests/all: $(TT_METAL_TESTS)
tt_metal/tests/%: $(TESTDIR)/tt_metal/tests/% ;

.PRECIOUS: $(TESTDIR)/tt_metal/tests/%
$(TESTDIR)/tt_metal/tests/%: $(OBJDIR)/tt_metal/tests/%.o $(BACKEND_LIB) $(TT_METAL_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_TESTS_LDFLAGS)

.PRECIOUS: $(TESTDIR)/tt_metal/tests/ops/%
$(TESTDIR)/tt_metal/tests/ops/%: $(OBJDIR)/tt_metal/tests/ops/%.o $(BACKEND_LIB) $(TT_METAL_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_metal/tests/%.o
$(OBJDIR)/tt_metal/tests/%.o: tt_metal/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -c -o $@ $<

.PRECIOUS: $(OBJDIR)/tt_metal/tests/ops/%.o
$(OBJDIR)/tt_metal/tests/ops/%.o: tt_metal/tests/ops/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -c -o $@ $<
