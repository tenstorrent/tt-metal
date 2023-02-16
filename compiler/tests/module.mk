# Every variable in subdir must be prefixed with subdir (emulating a namespace)
COMPILER_TESTS = \
	compiler/tests/test \
	compiler/tests/test_pybuda_graph \
	compiler/tests/test_graph_deserializer

COMPILER_TESTS_SRCS = $(addsuffix .cpp, $(COMPILER_TESTS))

COMPILER_TEST_INCLUDES = $(COMPILER_INCLUDES)
COMPILER_TESTS_LDFLAGS = -lgraph_utils -lgraph_deserializer -lreportify -lgraph_lib -lcommon -lstdc++fs -pthread -lyaml-cpp

COMPILER_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(COMPILER_TESTS_SRCS:.cpp=.o))
COMPILER_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(COMPILER_TESTS_SRCS:.cpp=.d))

-include $(COMPILER_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
compiler/tests: $(COMPILER_TESTS)
compiler/tests/all: $(COMPILER_TESTS)
compiler/tests/%: $(TESTDIR)/compiler/tests/% ;

.PRECIOUS: $(TESTDIR)/compiler/tests/%
$(TESTDIR)/compiler/tests/%: $(OBJDIR)/compiler/tests/%.o $(COMPILER_GRAPH_LIB) $(COMPILER_REPORTIFY) $(COMPILER_GRAPH_DESERIALIZER) $(COMPILER_GRAPH_UTILS)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(COMPILER_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(COMPILER_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/compiler/tests/%.o
$(OBJDIR)/compiler/tests/%.o: compiler/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(COMPILER_TEST_INCLUDES) -c -o $@ $<
