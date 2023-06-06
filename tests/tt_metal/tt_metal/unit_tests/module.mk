# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_UNIT_TESTS += \
	tests/dram/test_dram_to_l1_multicast.cpp \

TT_METAL_UNIT_TESTS_SRCS = $(addprefix tests/tt_metal/tt_metal/unit_tests/, $(addsuffix .cpp, $(TT_METAL_UNIT_TESTS:tests/%=%)))

TT_METAL_TEST_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES)
TT_METAL_UNIT_TESTS_LDFLAGS = -ltensor -ltt_dnn -ldtx -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp -ltt_dispatch

TT_METAL_UNIT_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_SRCS:.cpp=.o))
TT_METAL_UNIT_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_UNIT_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/tt_metal/unit_tests: $(TT_METAL_UNIT_TESTS)
tests/tt_metal/unit_tests/all: $(TT_METAL_UNIT_TESTS)
tests/tt_metal/unit_tests/%: $(TESTDIR)/tt_metal/% ;

.PRECIOUS: $(TESTDIR)/tt_metal/unit_tests/%
$(TESTDIR)/tt_metal/unit_tests/%: $(OBJDIR)/tt_metal/tests/unit_tests/%.o $(TT_METAL_LIB) $(TT_DNN_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_UNIT_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_metal/tests/unit_tests/%.o
$(OBJDIR)/tt_metal/tests/unit_tests/%.o: tests/tt_metal/tt_metal/unit_tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_TEST_INCLUDES) -c -o $@ $<
