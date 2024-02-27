# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_METAL_TESTS_HOME = tests/tt_metal/tt_metal
TT_METAL_GTEST_SMOKE_SRCS_HOME = tests/tt_metal/tt_metal/gtest_smoke

TT_METAL_GTEST_SMOKE = ${TT_METAL_GTEST_SMOKE_SRCS_HOME}/tests_main.cpp
TT_METAL_GTEST_SMOKE += $(wildcard ${TT_METAL_GTEST_SMOKE_SRCS_HOME}/*.cpp)
TT_METAL_GTEST_SMOKE += $(wildcard ${TT_METAL_GTEST_SMOKE_SRCS_HOME}/*/*.cpp)
TT_METAL_GTEST_SMOKE += $(wildcard ${TT_METAL_GTEST_SMOKE_SRCS_HOME}/*/*/*.cpp)

TT_METAL_GTEST_SMOKE_OBJ_HOME = tt_metal/tests/gtest_smoke/
TT_METAL_GTEST_SMOKE_SRCS = $(patsubst $(TT_METAL_GTEST_SMOKE_SRCS_HOME)%, $(TT_METAL_GTEST_SMOKE_OBJ_HOME)%, $(TT_METAL_GTEST_SMOKE))

TT_METAL_GTEST_SMOKE_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES)
TT_METAL_GTEST_SMOKE_LDFLAGS = $(LDFFLAGS) -ltt_metal -ldl -lstdc++fs -pthread -lyaml-cpp -lgtest -lgtest_main -lm

TT_METAL_GTEST_SMOKE_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_GTEST_SMOKE_SRCS:.cpp=.o))
TT_METAL_GTEST_SMOKE_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_GTEST_SMOKE_SRCS:.cpp=.d))

-include $(TT_METAL_GTEST_SMOKE_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/tt_metal/gtest_smoke: $(TESTDIR)/tt_metal/gtest_smoke

.PRECIOUS: $(TESTDIR)/tt_metal/gtest_smoke
$(TESTDIR)/tt_metal/gtest_smoke: $(TT_METAL_GTEST_SMOKE_OBJS) $(TT_METAL_LIB) $(TT_DNN_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_GTEST_SMOKE_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_METAL_GTEST_SMOKE_LDFLAGS)

.PRECIOUS: $(OBJDIR)/$(TT_METAL_GTEST_SMOKE_OBJ_HOME)/%.o
$(OBJDIR)/$(TT_METAL_GTEST_SMOKE_OBJ_HOME)/%.o: $(TT_METAL_GTEST_SMOKE_SRCS_HOME)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_METAL_GTEST_SMOKE_INCLUDES) -c -o $@ $<
