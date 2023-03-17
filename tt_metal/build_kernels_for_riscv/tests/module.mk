# Every variable in subdir must be prefixed with subdir (emulating a namespace)
BUILD_KERNELS_FOR_RISCV_TESTS = \
	build_kernels_for_riscv/tests/test_build_kernel_add_two_ints \
	build_kernels_for_riscv/tests/test_build_kernel_dram_copy \
	build_kernels_for_riscv/tests/test_build_kernel_dram_copy_ncrisc \
	build_kernels_for_riscv/tests/test_build_kernel_dram_copy_brisc_ncrisc \
	build_kernels_for_riscv/tests/test_build_kernel_dram_copy_looped \
	build_kernels_for_riscv/tests/test_build_kernel_blank \
	build_kernels_for_riscv/tests/test_build_kernel_datacopy \
	build_kernels_for_riscv/tests/test_build_kernel_datacopy_switched_riscs \
	build_kernels_for_riscv/tests/test_build_kernel_dram_to_l1_copy \
	build_kernels_for_riscv/tests/test_build_kernel_l1_to_dram_copy \
	build_kernels_for_riscv/tests/test_build_kernel_copy_pattern \
	build_kernels_for_riscv/tests/test_build_kernel_copy_pattern_tilized \
	build_kernels_for_riscv/tests/test_build_kernel_loader_sync \
	build_kernels_for_riscv/tests/test_build_kernel_loader_sync_db \
	build_kernels_for_riscv/tests/test_build_kernel_eltwise_sync \
	build_kernels_for_riscv/tests/test_build_kernel_remote_read_remote_write_sync \
	build_kernels_for_riscv/tests/test_build_kernel_remote_read_remote_write_sync_db \
	build_kernels_for_riscv/tests/test_build_kernel_risc_read_speed \
	build_kernels_for_riscv/tests/test_build_kernel_risc_write_speed \
	build_kernels_for_riscv/tests/test_build_kernel_transpose_hc \
	build_kernels_for_riscv/tests/test_build_kernel_test_debug_print \
	build_kernels_for_riscv/tests/test_build_kernel_risc_rw_speed_banked_dram \
	build_kernels_for_riscv/tests/test_build_kernel_matmul_small_block \
	build_kernels_for_riscv/tests/test_build_kernel_dataflow_cb_test

BUILD_KERNELS_FOR_RISCV_TESTS_SRCS = $(addprefix tt_metal/, $(addsuffix .cpp, $(BUILD_KERNELS_FOR_RISCV_TESTS)))

BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES = $(BUILD_KERNELS_FOR_RISCV_INCLUDES) -I$(TT_METAL_HOME)/build_kernels_for_riscv/tests
BUILD_KERNELS_FOR_RISCV_TESTS_LDFLAGS = -lpthread -lstdc++fs -lbuild_kernels_for_riscv -lhlkc_api -lcommon -lyaml-cpp -ldl

BUILD_KERNELS_FOR_RISCV_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_TESTS_SRCS:.cpp=.o))
BUILD_KERNELS_FOR_RISCV_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_TESTS_SRCS:.cpp=.d))

-include $(BUILD_KERNELS_FOR_RISCV_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
build_kernels_for_riscv/tests: $(BUILD_KERNELS_FOR_RISCV_TESTS)
build_kernels_for_riscv/tests/all: $(BUILD_KERNELS_FOR_RISCV_TESTS)
build_kernels_for_riscv/tests/%: $(TESTDIR)/build_kernels_for_riscv/tests/% ;

.PRECIOUS: $(TESTDIR)/build_kernels_for_riscv/tests/%
$(TESTDIR)/build_kernels_for_riscv/tests/%: $(OBJDIR)/build_kernels_for_riscv/tests/%.o  $(OP_LIB) $(CORE_GRAPH_LIB) $(BUILD_KERNELS_FOR_RISCV_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(BUILD_KERNELS_FOR_RISCV_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/build_kernels_for_riscv/tests/%.o
$(OBJDIR)/build_kernels_for_riscv/tests/%.o: build_kernels_for_riscv/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES) -c -o $@ $<
