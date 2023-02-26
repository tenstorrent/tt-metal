# Every variable in subdir must be prefixed with subdir (emulating a namespace)
COMPILER_GRAPH_LIB = $(LIBDIR)/libgraph_lib.a
COMPILER_GRAPH_LIB_SRCS = \
	compiler/graph_lib/edge.cpp \
	compiler/graph_lib/graph.cpp \
	compiler/graph_lib/node.cpp \
	compiler/graph_lib/node_types.cpp \
	compiler/graph_lib/shape.cpp \
	compiler/graph_lib/utils.cpp \
	compiler/graph_lib/common.cpp

COMPILER_GRAPH_LIB_INCLUDES = $(COMMON_INCLUDES) $(COMPILER_INCLUDES) 

COMPILER_GRAPH_LIB_OBJS = $(addprefix $(OBJDIR)/, $(COMPILER_GRAPH_LIB_SRCS:.cpp=.o))
COMPILER_GRAPH_LIB_DEPS = $(addprefix $(OBJDIR)/, $(COMPILER_GRAPH_LIB_SRCS:.cpp=.d))

-include $(COMPILER_GRAPH_LIB_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
compiler/graph_lib: $(COMPILER_GRAPH_LIB)

$(COMPILER_GRAPH_LIB): $(COMMON_LIB) $(COMPILER_GRAPH_LIB_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/compiler/graph_lib/%.o: compiler/graph_lib/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMPILER_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMPILER_GRAPH_LIB_INCLUDES) -c -o $@ $<

# We need to include at top level for now because the tests depend on model
#include graph_lib/tests/module.mk
