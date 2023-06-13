# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_UNIT_TESTS += \
	tests_main.cpp \
	basic/basic.cpp \
	compute/basic_sfpu_compute.cpp \
	dram/single_core_dram.cpp


TT_METAL_UNIT_TESTS_HOME = tt_metal/tests/unit_tests/
TT_METAL_UNIT_TESTS_SRCS = $(addprefix $(TT_METAL_UNIT_TESTS_HOME), $(TT_METAL_UNIT_TESTS:%=%))

TT_METAL_UNIT_TESTS_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES) -I$(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests/common
TT_METAL_UNIT_TESTS_LDFLAGS = -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp -lgtest

TT_METAL_UNIT_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_SRCS:.cpp=.o))
TT_METAL_UNIT_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_UNIT_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/tt_metal/unit_tests: $(TESTDIR)/tt_metal/unit_tests

.PRECIOUS: $(TESTDIR)/tt_metal/unit_tests
$(TESTDIR)/tt_metal/unit_tests: $(TT_METAL_UNIT_TESTS_OBJS) $(TT_METAL_LIB) $(TT_DNN_LIB) $(GTEST_LIBRARIES)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_UNIT_TESTS_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_UNIT_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_metal/tests/unit_tests/%.o
$(OBJDIR)/tt_metal/tests/unit_tests/%.o: tests/tt_metal/tt_metal/unit_tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_UNIT_TESTS_INCLUDES) -c -o $@ $<
