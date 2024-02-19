# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TTNN_UNIT_TESTS_HOME_DIR = $(TT_METAL_HOME)/tests/ttnn/unit_tests

TTNN_UNIT_TESTS_DIRS := $(TTNN_UNIT_TESTS_HOME_DIR) $(TTNN_UNIT_TESTS_HOME_DIR)/operations

TTNN_UNIT_TESTS_SRCS := $(foreach dir,$(TTNN_UNIT_TESTS_DIRS),$(wildcard $(dir)/*.cpp))

TTNN_UNIT_TESTS_INCLUDES := $(TEST_INCLUDES) $(TT_METAL_INCLUDES)

TTNN_UNIT_TESTS_LDFLAGS := -lttnn -ltt_dnn -ltt_metal -ldl -lstdc++fs -pthread -lyaml-cpp -lgtest -lgtest_main -lm

TTNN_UNIT_TESTS_OBJS := $(addprefix $(OBJDIR)/, $(TTNN_UNIT_TESTS_SRCS:$(TTNN_UNIT_TESTS_HOME_DIR)/%.cpp=ttnn/tests/unit_tests/%.o))
TTNN_UNIT_TESTS_DEPS := $(TTNN_UNIT_TESTS_OBJS:.o=.d)

-include $(TTNN_UNIT_TESTS_DEPS)

tests/ttnn/unit_tests: $(TESTDIR)/ttnn/unit_tests

.PRECIOUS: $(OBJDIR)/ttnn/tests/%.o
$(OBJDIR)/ttnn/tests/%.o: $(TT_METAL_HOME)/tests/ttnn/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TTNN_UNIT_TESTS_INCLUDES) -c -o $@ $<

.PHONY: tests/ttnn/unit_tests_run
tests/ttnn/unit_tests_run: tests/ttnn/unit_tests
	@echo "Running all ttnn unit tests"

.PRECIOUS: $(TESTDIR)/ttnn/unit_tests
$(TESTDIR)/ttnn/unit_tests: $(TTNN_UNIT_TESTS_OBJS) $(TT_METAL_LIB) $(TT_DNN_LIB) $(TTNN_LIB)
	$(info [${TTNN_UNIT_TESTS_OBJS}])
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(TTNN_UNIT_TESTS_LDFLAGS)
