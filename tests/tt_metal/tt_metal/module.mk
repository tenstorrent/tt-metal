include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests_common/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests_fast_dispatch/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests_fast_dispatch_single_chip_multi_queue/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests_frequent/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/gtest_smoke/module.mk

# Programming examples for external users
include $(TT_METAL_HOME)/tt_metal/programming_examples/module.mk

# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_TESTS += \
		 tests/tt_metal/test_bmm \
		 tests/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch \
		 tests/tt_metal/perf_microbenchmark/dispatch/test_bw_and_latency \
		 tests/tt_metal/perf_microbenchmark/dispatch/test_dispatcher \
		 tests/tt_metal/perf_microbenchmark/dispatch/test_prefetcher \
		 tests/tt_metal/perf_microbenchmark/ethernet/test_ethernet_read_and_send_data \
		 tests/tt_metal/perf_microbenchmark/ethernet/test_workers_and_erisc_datamover_unidirectional \
		 tests/tt_metal/perf_microbenchmark/routing/test_tx_rx \
		 tests/tt_metal/perf_microbenchmark/routing/test_mux_demux \
		 tests/tt_metal/perf_microbenchmark/routing/test_mux_demux_2level \
		 tests/tt_metal/perf_microbenchmark/routing/test_uni_tunnel \
		 tests/tt_metal/perf_microbenchmark/routing/test_uni_tunnel_single_chip \
		 tests/tt_metal/perf_microbenchmark/routing/test_bi_tunnel \
		 tests/tt_metal/perf_microbenchmark/noc/test_noc_unicast_vs_multicast_to_single_core_latency \
		 tests/tt_metal/perf_microbenchmark/old/matmul/matmul_global_l1 \
		 tests/tt_metal/perf_microbenchmark/old/matmul/matmul_local_l1 \
		 tests/tt_metal/perf_microbenchmark/old/noc/test_noc_read_global_l1 \
		 tests/tt_metal/perf_microbenchmark/old/noc/test_noc_read_local_l1 \
		 tests/tt_metal/perf_microbenchmark/old/pcie/test_enqueue_rw_buffer \
		 tests/tt_metal/perf_microbenchmark/old/pcie/test_rw_buffer \
		 tests/tt_metal/perf_microbenchmark/old/pcie/test_rw_device_dram \
		 tests/tt_metal/perf_microbenchmark/old/pcie/test_rw_device_l1 \
		 tests/tt_metal/perf_microbenchmark/1_compute_mm/test_compute_mm \
		 tests/tt_metal/perf_microbenchmark/2_noc_adjacent/test_noc_adjacent \
		 tests/tt_metal/perf_microbenchmark/2_noc_rtor/test_noc_rtor \
		 tests/tt_metal/perf_microbenchmark/3_pcie_transfer/test_rw_buffer \
		 tests/tt_metal/perf_microbenchmark/6_dram_offchip/test_dram_offchip \
		 tests/tt_metal/perf_microbenchmark/7_kernel_launch/test_kernel_launch \
		 tests/tt_metal/perf_microbenchmark/noc/test_noc_unicast_vs_multicast_to_single_core_latency \
		 tests/tt_metal/test_add_two_ints \
		 tests/tt_metal/test_compile_args \
		 tests/tt_metal/test_eltwise_binary \
		 tests/tt_metal/test_eltwise_unary \
		 tests/tt_metal/test_matmul_single_tile_bfp8b \
		 tests/tt_metal/test_matmul_single_tile_output_in_l1 \
		 tests/tt_metal/test_dram_loopback_single_core \
		 tests/tt_metal/test_datacopy_bfp8b \
		 tests/tt_metal/test_datacopy \
		 tests/tt_metal/test_datacopy_output_in_l1 \
		 tests/tt_metal/test_dataflow_cb \
		 tests/tt_metal/test_transpose_hc \
		 tests/tt_metal/test_transpose_wh \
		 tests/tt_metal/test_multiple_programs \
		 tests/tt_metal/test_multi_core_kernel \
		 tests/tt_metal/test_unpack_tilize \
		 tests/tt_metal/test_unpack_untilize \
		 tests/tt_metal/test_interleaved_layouts \
		 tests/tt_metal/test_interleaved_l1_buffer \
		 tests/tt_metal/test_bcast \
		 tests/tt_metal/test_generic_binary_reader_matmul_large_block \
		 tests/tt_metal/test_3x3conv_as_matmul_large_block \
		 tests/tt_metal/test_l1_to_l1_multi_core \
		 tests/tt_metal/test_dram_copy_sticks_multi_core \
		 tests/tt_metal/test_reduce_h \
		 tests/tt_metal/test_reduce_w \
		 tests/tt_metal/test_reduce_hw \
		 tests/tt_metal/test_untilize_eltwise_binary \
		 tests/tt_metal/test_bfp8_conversion \
		 tests/tt_metal/test_bfp4_conversion \
		 tests/tt_metal/tt_dispatch/test_enqueue_program \
		 tests/tt_metal/test_core_range_set \
		 tests/tt_metal/test_compile_sets_kernel_binaries \
		 tests/tt_metal/test_compile_program \
		 # test/tt_metal/test_datacopy_multi_core_multi_dram \  # this does not compile
		 # tests/tt_metal/test_dram_to_l1_multicast \ # these tests have all been converted to gtest
		 # tests/tt_metal/test_dram_to_l1_multicast_loopback_src \
		 # tests/tt_metal/test_dram_loopback_single_core_db \
		 # tests/tt_metal/test_matmul_multi_tile \
		 # tests/tt_metal/test_matmul_large_block \
		 # tests/tt_metal/test_matmul_single_core \
		 # tests/tt_metal/test_matmul_single_core_small \
		 # tests/tt_metal/test_matmul_multi_core_single_dram \
		 # tests/tt_metal/test_matmul_multi_core_multi_dram \
		 # tests/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast \
		 # tests/tt_metal/test_matmul_multi_core_multi_dram_in1_mcast \
		 # tests/tt_metal/test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast \
		 # tests/tt_metal/test_matmul_single_tile \
		 # tests/tt_metal/test_flatten \

TT_METAL_TESTS_SRCS = $(addprefix tests/tt_metal/, $(addsuffix .cpp, $(TT_METAL_TESTS:tests/%=%)))

TT_METAL_TESTS_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES)
TT_METAL_TESTS_LDFLAGS = $(LDFLAGS) -ltt_metal -ldl -lstdc++fs -pthread -lyaml-cpp -lgtest -lm

TT_METAL_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.o))
TT_METAL_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/tt_metal: $(TT_METAL_TESTS) programming_examples tests/tt_metal/gtest_smoke tests/tt_metal/unit_tests tests/tt_metal/unit_tests_fast_dispatch tests/tt_metal/unit_tests_fast_dispatch_single_chip_multi_queue tests/tt_metal/unit_tests_frequent
tests/tt_metal/all: $(TT_METAL_TESTS)
tests/tt_metal/%: $(TESTDIR)/tt_metal/% ;

.PRECIOUS: $(TESTDIR)/tt_metal/%
$(TESTDIR)/tt_metal/%: $(OBJDIR)/tt_metal/tests/%.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TESTS_INCLUDES) -o $@ $^ $(TT_METAL_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_metal/tests/%.o
$(OBJDIR)/tt_metal/tests/%.o: tests/tt_metal/tt_metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TESTS_INCLUDES) -c -o $@ $<
