COMPILER_GRAPH_UTILS = ${LIBDIR}/libgraph_utils.a
COMPILER_GRAPH_UTILS_SRCS = \
	compiler/graph_utils/op_reducer.cpp \
	compiler/graph_utils/op_histogram.cpp

COMPILER_GRAPH_UTILS_OBJS = $(addprefix $(OBJDIR)/, $(COMPILER_GRAPH_UTILS_SRCS:.cpp=.o))
COMPILER_GRAPH_UTILS_DEPS = $(addprefix $(OBJDIR)/, $(COMPILER_GRAPH_UTILS_SRCS:.cpp=.d))

COMPILER_GRAPH_UTILS_INCLUDES = $(COMPILER_INCLUDES)
COMPILER_GRAPH_UTILS_LDFLAGS = -lreportify -lgraph_lib -lcommon

-include $(COMPILER_GRAPH_UTILS_DEPS)

compiler/graph_utils: $(COMPILER_GRAPH_UTILS)

$(COMPILER_GRAPH_UTILS): $(COMPILER_GRAPH_LIB) $(COMPILER_REPORTIFY) $(COMPILER_GRAPH_UTILS_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/compiler/graph_utils/%.o: compiler/graph_utils/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMPILER_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMPILER_GRAPH_UTILS_INCLUDES) -c -o $@ $<
