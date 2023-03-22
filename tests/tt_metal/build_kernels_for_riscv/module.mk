# Every variable in subdir must be prefixed with subdir (emulating a namespace)
BUILD_KERNELS_FOR_RISCV_TESTS = \
	tests/build_kernels_for_riscv/test_build_kernel_add_two_ints \
	tests/build_kernels_for_riscv/test_build_kernel_dram_copy \
	tests/build_kernels_for_riscv/test_build_kernel_dram_copy_ncrisc \
	tests/build_kernels_for_riscv/test_build_kernel_dram_copy_brisc_ncrisc \
	tests/build_kernels_for_riscv/test_build_kernel_dram_copy_looped \
	tests/build_kernels_for_riscv/test_build_kernel_blank \
	tests/build_kernels_for_riscv/test_build_kernel_datacopy \
	tests/build_kernels_for_riscv/test_build_kernel_datacopy_switched_riscs \
	tests/build_kernels_for_riscv/test_build_kernel_dram_to_l1_copy \
	tests/build_kernels_for_riscv/test_build_kernel_l1_to_dram_copy \
	tests/build_kernels_for_riscv/test_build_kernel_copy_pattern \
	tests/build_kernels_for_riscv/test_build_kernel_copy_pattern_tilized \
	tests/build_kernels_for_riscv/test_build_kernel_loader_sync \
	tests/build_kernels_for_riscv/test_build_kernel_loader_sync_db \
	tests/build_kernels_for_riscv/test_build_kernel_eltwise_sync \
	tests/build_kernels_for_riscv/test_build_kernel_remote_read_remote_write_sync \
	tests/build_kernels_for_riscv/test_build_kernel_remote_read_remote_write_sync_db \
	tests/build_kernels_for_riscv/test_build_kernel_risc_read_speed \
	tests/build_kernels_for_riscv/test_build_kernel_risc_write_speed \
	tests/build_kernels_for_riscv/test_build_kernel_transpose_hc \
	tests/build_kernels_for_riscv/test_build_kernel_test_debug_print \
	tests/build_kernels_for_riscv/test_build_kernel_risc_rw_speed_banked_dram \
	tests/build_kernels_for_riscv/test_build_kernel_matmul_small_block \
	tests/build_kernels_for_riscv/test_build_kernel_dataflow_cb_test

BUILD_KERNELS_FOR_RISCV_TESTS_SRCS = $(addprefix tests/tt_metal/, $(addsuffix .cpp, $(BUILD_KERNELS_FOR_RISCV_TESTS:tests/%:%)))

BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES = $(TEST_INCLUDES) $(BUILD_KERNELS_FOR_RISCV_INCLUDES)
BUILD_KERNELS_FOR_RISCV_TESTS_LDFLAGS = -lpthread -lstdc++fs -lbuild_kernels_for_riscv -lhlkc_api -lcommon -lyaml-cpp -ldl

BUILD_KERNELS_FOR_RISCV_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_TESTS_SRCS:.cpp=.o))
BUILD_KERNELS_FOR_RISCV_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_TESTS_SRCS:.cpp=.d))

-include $(BUILD_KERNELS_FOR_RISCV_TESTS_DEPS)

tests/build_kernels_for_riscv: $(BUILD_KERNELS_FOR_RISCV_TESTS)
tests/build_kernels_for_riscv/all: $(BUILD_KERNELS_FOR_RISCV_TESTS)
tests/build_kernels_for_riscv/test_%: $(TESTDIR)/build_kernels_for_riscv/test_%;

.PRECIOUS: $(TESTDIR)/build_kernels_for_riscv/test_%
$(TESTDIR)/build_kernels_for_riscv/test_%: $(OBJDIR)/build_kernels_for_riscv/tests/test_%.o  $(OP_LIB) $(CORE_GRAPH_LIB) $(BUILD_KERNELS_FOR_RISCV_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(BUILD_KERNELS_FOR_RISCV_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/build_kernels_for_riscv/tests/test_%.o
$(OBJDIR)/build_kernels_for_riscv/tests/test_%.o: tests/tt_metal/build_kernels_for_riscv/test_%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(BUILD_KERNELS_FOR_RISCV_TEST_INCLUDES) -c -o $@ $<

# To support older workflows, should delete later
build_kernels_for_riscv/tests: tests/build_kernels_for_riscv
build_kernels_for_riscv/tests/all: tests/build_kernels_for_riscv/all
