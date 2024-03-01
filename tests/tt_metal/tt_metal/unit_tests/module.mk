# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_METAL_UNIT_TESTS_SRCS_HOME = tests/tt_metal/tt_metal/unit_tests

TT_METAL_UNIT_TESTS = ${TT_METAL_UNIT_TESTS_SRCS_HOME}/tests_main.cpp
TT_METAL_UNIT_TESTS += $(wildcard ${TT_METAL_UNIT_TESTS_SRCS_HOME}/*/*.cpp)
TT_METAL_UNIT_TESTS += $(wildcard ${TT_METAL_UNIT_TESTS_SRCS_HOME}/*/*/*.cpp)

TT_METAL_UNIT_TESTS_OBJ_HOME = tt_metal/tests/unit_tests/
TT_METAL_UNIT_TESTS_SRCS = $(patsubst $(TT_METAL_UNIT_TESTS_SRCS_HOME)%, $(TT_METAL_UNIT_TESTS_OBJ_HOME)%, $(TT_METAL_UNIT_TESTS))

TT_METAL_UNIT_TESTS_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES) -I$(TT_METAL_HOME)/tests/tt_metal/tt_metal/unit_tests/common
TT_METAL_UNIT_TESTS_LDFLAGS = $(TT_METAL_TESTS_LDFLAGS) -ltt_metal -ldl -lstdc++fs -pthread -lyaml-cpp -lgtest -lgtest_main -lm

TT_METAL_UNIT_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_SRCS:.cpp=.o))
TT_METAL_UNIT_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_SRCS:.cpp=.d))

-include $(TT_METAL_UNIT_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/tt_metal/unit_tests: $(TESTDIR)/tt_metal/unit_tests

.PRECIOUS: $(TESTDIR)/tt_metal/unit_tests
$(TESTDIR)/tt_metal/unit_tests: $(TT_METAL_UNIT_TESTS_OBJS) $(TT_METAL_UNIT_TESTS_COMMON_OBJS) $(TT_METAL_LIB) $(TT_DNN_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_UNIT_TESTS_INCLUDES) -o $@ $^ $(TT_METAL_UNIT_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/$(TT_METAL_UNIT_TESTS_OBJ_HOME)/%.o
$(OBJDIR)/$(TT_METAL_UNIT_TESTS_OBJ_HOME)/%.o: $(TT_METAL_UNIT_TESTS_SRCS_HOME)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_UNIT_TESTS_INCLUDES) -c -o $@ $<
