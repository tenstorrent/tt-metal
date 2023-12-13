# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_METAL_UNIT_TESTS_COMMON_SRCS_HOME = tests/tt_metal/tt_metal/unit_tests_common

TT_METAL_UNIT_TESTS_COMMON = $(wildcard ${TT_METAL_UNIT_TESTS_COMMON_SRCS_HOME}/*/*.cpp)
TT_METAL_UNIT_TESTS_COMMON += $(wildcard ${TT_METAL_UNIT_TESTS_COMMON_SRCS_HOME}/*/*/*.cpp)

TT_METAL_UNIT_TESTS_COMMON_OBJ_HOME = tt_metal/tests/unit_tests_common/
TT_METAL_UNIT_TESTS_COMMON_SRCS = $(patsubst $(TT_METAL_UNIT_TESTS_COMMON_SRCS_HOME)%, $(TT_METAL_UNIT_TESTS_COMMON_OBJ_HOME)%, $(TT_METAL_UNIT_TESTS_COMMON))

TT_METAL_UNIT_TESTS_COMMON_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES) -I$(TT_METAL_HOME)/$(TT_METAL_UNIT_TESTS_COMMON_SRCS_HOME)/common

TT_METAL_UNIT_TESTS_COMMON_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_COMMON_SRCS:.cpp=.o))
TT_METAL_UNIT_TESTS_COMMON_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_UNIT_TESTS_COMMON_SRCS:.cpp=.d))

-include $(TT_METAL_UNIT_TESTS_COMMON_DEPS)

# This module doesn't build as its own executable, it's just included in unit_tests and unit_tests_fast_dispatch
.PRECIOUS: $(OBJDIR)/$(TT_METAL_UNIT_TESTS_COMMON_OBJ_HOME)/%.o
$(OBJDIR)/$(TT_METAL_UNIT_TESTS_COMMON_OBJ_HOME)/%.o: $(TT_METAL_UNIT_TESTS_COMMON_SRCS_HOME)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_UNIT_TESTS_COMMON_INCLUDES) -c -o $@ $<
