COMPILER_REPORTIFY = ${LIBDIR}/libreportify.a
COMPILER_REPORTIFY_SRCS = \
	compiler/reportify/reportify.cpp \
	compiler/reportify/paths.cpp

COMPILER_REPORTIFY_OBJS = $(addprefix $(OBJDIR)/, $(COMPILER_REPORTIFY_SRCS:.cpp=.o))
PYBUDA_CSRC_REPORTIFY_DEPS = $(addprefix $(OBJDIR)/, $(COMPILER_REPORTIFY_SRCS:.cpp=.d))

COMPILER_REPORTIFY_INCLUDES = $(COMPILER_INCLUDES)

-include $(COMPILER_REPORTIFY_DEPS)

compiler/reportify: $(COMPILER_REPORTIFY)

$(COMPILER_REPORTIFY): $(COMPILER_GRAPH_LIB) $(COMPILER_REPORTIFY_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/compiler/reportify/%.o: compiler/reportify/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMPILER_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMPILER_REPORTIFY_INCLUDES) -c -o $@ $<

