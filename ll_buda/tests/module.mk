# Every variable in subdir must be prefixed with subdir (emulating a namespace)
LL_BUDA_TESTS += \
		 ll_buda/tests/ops/test_eltwise_binary_op \
		 ll_buda/tests/ops/test_eltwise_unary_op \
		 ll_buda/tests/dtx/unit_tests \
		 ll_buda/tests/dtx/tensor \
		 ll_buda/tests/dtx/overlap \
		 ll_buda/tests/dtx/collapse_transformations \
		 ll_buda/tests/ops/test_transpose_op \
		 ll_buda/tests/ops/test_reduce_op \
		 ll_buda/tests/ops/test_bcast_op \
		 ll_buda/tests/ops/test_bmm_op \
		 ll_buda/tests/test_bmm \
		 ll_buda/tests/tensors/test_host_device_loopback \
		 ll_buda/tests/test_add_two_ints \
		 ll_buda/tests/test_dram_to_l1_multicast \
		 ll_buda/tests/test_dram_to_l1_multicast_loopback_src \
		 ll_buda/tests/test_dram_loopback_multi_core \
		 ll_buda/tests/test_dram_loopback_multi_core_db \
		 ll_buda/tests/test_dram_loopback_single_core \
		 ll_buda/tests/test_dram_loopback_single_core_db \
		 ll_buda/tests/test_eltwise_binary \
		 ll_buda/tests/test_matmul_single_tile \
		 ll_buda/tests/test_matmul_multi_tile \
		 ll_buda/tests/test_matmul_large_block \
		 ll_buda/tests/test_matmul_single_core \
		 ll_buda/tests/test_matmul_multi_core_single_dram \
		 ll_buda/tests/test_matmul_multi_core_multi_dram \
		 ll_buda/tests/test_matmul_multi_core_multi_dram_in0_mcast \
		 ll_buda/tests/test_matmul_multi_core_multi_dram_in1_mcast \
		 ll_buda/tests/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast \
		 ll_buda/tests/test_datacopy \
		 ll_buda/tests/test_dataflow_cb \
		 ll_buda/tests/test_flatten \
		 ll_buda/tests/test_transpose_hc \
		 ll_buda/tests/test_reduce_h \
		 ll_buda/tests/test_reduce_w \
		 ll_buda/tests/test_reduce_hw \
		 ll_buda/tests/test_transpose_wh \
		 ll_buda/tests/test_sfpu \
		 ll_buda/tests/test_multiple_programs \
		 ll_buda/tests/test_multi_core_kernel \
		 ll_buda/tests/test_unpack_tilize \
		 ll_buda/tests/test_unpack_untilize \
		 ll_buda/tests/test_graph_interpreter \
		 ll_buda/tests/test_bcast \
		 ll_buda/tests/test_interleaved_layouts \
		 ll_buda/tests/test_3x3conv_as_matmul_large_block \
		 ll_buda/tests/test_generic_binary_reader_matmul_large_block \
		 ll_buda/tests/test_dtx \
		 ll_buda/tests/test_dtx_tilized_row_to_col_major \
		 ll_buda/tests/test_pipeline_across_rows \

LL_BUDA_TESTS_SRCS = $(addsuffix .cpp, $(LL_BUDA_TESTS))

LL_BUDA_TEST_INCLUDES = $(LL_BUDA_INCLUDES) -Itt_gdb -Ill_buda/tests -Icompile_trisc -Iverif
LL_BUDA_TESTS_LDFLAGS = -lll_buda_impl -lll_buda -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -lhlkc_api -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

LL_BUDA_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(LL_BUDA_TESTS_SRCS:.cpp=.o))
LL_BUDA_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(LL_BUDA_TESTS_SRCS:.cpp=.d))

-include $(LL_BUDA_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
ll_buda/tests: $(LL_BUDA_TESTS)
ll_buda/tests/all: $(LL_BUDA_TESTS)
ll_buda/tests/%: $(TESTDIR)/ll_buda/tests/% ;

.PRECIOUS: $(TESTDIR)/ll_buda/tests/%
$(TESTDIR)/ll_buda/tests/%: $(OBJDIR)/ll_buda/tests/%.o $(BACKEND_LIB) $(LL_BUDA_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LL_BUDA_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(LL_BUDA_TESTS_LDFLAGS)

.PRECIOUS: $(TESTDIR)/ll_buda/tests/ops/%
$(TESTDIR)/ll_buda/tests/ops/%: $(OBJDIR)/ll_buda/tests/ops/%.o $(BACKEND_LIB) $(LL_BUDA_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LL_BUDA_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(LL_BUDA_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/ll_buda/tests/%.o
$(OBJDIR)/ll_buda/tests/%.o: ll_buda/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LL_BUDA_TEST_INCLUDES) -c -o $@ $<

.PRECIOUS: $(OBJDIR)/ll_buda/tests/ops/%.o
$(OBJDIR)/ll_buda/tests/ops/%.o: ll_buda/tests/ops/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LL_BUDA_TEST_INCLUDES) -c -o $@ $<
