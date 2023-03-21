# Every variable in subdir must be prefixed with subdir (emulating a namespace)
BUILD_KERNELS_FOR_RISCV_TESTS = \
	test/build_kernels_for_riscv/test_build_kernel_add_two_ints \
	test/build_kernels_for_riscv/test_build_kernel_dram_copy \
	test/build_kernels_for_riscv/test_build_kernel_dram_copy_ncrisc \
	test/build_kernels_for_riscv/test_build_kernel_dram_copy_brisc_ncrisc \
	test/build_kernels_for_riscv/test_build_kernel_dram_copy_looped \
	test/build_kernels_for_riscv/test_build_kernel_blank \
	test/build_kernels_for_riscv/test_build_kernel_datacopy \
	test/build_kernels_for_riscv/test_build_kernel_datacopy_switched_riscs \
	test/build_kernels_for_riscv/test_build_kernel_dram_to_l1_copy \
	test/build_kernels_for_riscv/test_build_kernel_l1_to_dram_copy \
	test/build_kernels_for_riscv/test_build_kernel_copy_pattern \
	test/build_kernels_for_riscv/test_build_kernel_copy_pattern_tilized \
	test/build_kernels_for_riscv/test_build_kernel_loader_sync \
	test/build_kernels_for_riscv/test_build_kernel_loader_sync_db \
	test/build_kernels_for_riscv/test_build_kernel_eltwise_sync \
	test/build_kernels_for_riscv/test_build_kernel_remote_read_remote_write_sync \
	test/build_kernels_for_riscv/test_build_kernel_remote_read_remote_write_sync_db \
	test/build_kernels_for_riscv/test_build_kernel_risc_read_speed \
	test/build_kernels_for_riscv/test_build_kernel_risc_write_speed \
	test/build_kernels_for_riscv/test_build_kernel_transpose_hc \
	test/build_kernels_for_riscv/test_build_kernel_test_debug_print \
	test/build_kernels_for_riscv/test_build_kernel_risc_rw_speed_banked_dram \
	test/build_kernels_for_riscv/test_build_kernel_matmul_small_block \
	test/build_kernels_for_riscv/test_build_kernel_dataflow_cb_test

BUILD_KERNELS_FOR_RISCV_TESTS_SRCS = $(addprefix test/tt_metal/, $(addsuffix .cpp, $(BUILD_KERNELS_FOR_RISCV_TESTS:test/%:%)))

BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES = $(TEST_INCLUDES) $(BUILD_KERNELS_FOR_RISCV_INCLUDES) -I$(TT_METAL_HOME)/build_kernels_for_riscv/tests
BUILD_KERNELS_FOR_RISCV_TESTS_LDFLAGS = -lpthread -lstdc++fs -lbuild_kernels_for_riscv -lhlkc_api -lcommon -lyaml-cpp -ldl

BUILD_KERNELS_FOR_RISCV_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_TESTS_SRCS:.cpp=.o))
BUILD_KERNELS_FOR_RISCV_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_TESTS_SRCS:.cpp=.d))

-include $(BUILD_KERNELS_FOR_RISCV_TESTS_DEPS)

test/build_kernels_for_riscv: $(BUILD_KERNELS_FOR_RISCV_TESTS)
test/build_kernels_for_riscv/all: $(BUILD_KERNELS_FOR_RISCV_TESTS)
test/build_kernels_for_riscv/test_%: $(TESTDIR)/build_kernels_for_riscv/test_%;

.PRECIOUS: $(TESTDIR)/build_kernels_for_riscv/test_%
$(TESTDIR)/build_kernels_for_riscv/test_%: $(OBJDIR)/build_kernels_for_riscv/test/test_%.o  $(OP_LIB) $(CORE_GRAPH_LIB) $(BUILD_KERNELS_FOR_RISCV_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(BUILD_KERNELS_FOR_RISCV_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/build_kernels_for_riscv/test/test_%.o
$(OBJDIR)/build_kernels_for_riscv/test/test_%.o: test/tt_metal/build_kernels_for_riscv/test_%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES) -c -o $@ $<

# To support older workflows, should delete later
build_kernels_for_riscv/tests: test/build_kernels_for_riscv
build_kernels_for_riscv/tests/all: test/build_kernels_for_riscv/all
