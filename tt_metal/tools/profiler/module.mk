# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PROFILER_INCLUDES = $(BASE_INCLUDES)

PROFILER_LIB = $(LIBDIR)/tools/libprofiler.a
PROFILER_DEFINES =
PROFILER_INCLUDES += -Itools/profiler
PROFILER_CFLAGS = $(CFLAGS) -Werror

PROFILER_SRCS += \
	$(wildcard tt_metal/tools/profiler/*.cpp)

PROFILER_OBJS = $(addprefix $(OBJDIR)/, $(PROFILER_SRCS:.cpp=.o))
PROFILER_DEPS = $(addprefix $(OBJDIR)/, $(PROFILER_SRCS:.cpp=.d))

-include $(PROFILER_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tools/profiler: $(PROFILER_LIB)

$(PROFILER_LIB): $(PROFILER_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(PROFILER_OBJS)

$(OBJDIR)/tt_metal/tools/profiler/%.o: tt_metal/tools/profiler/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PROFILER_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PROFILER_INCLUDES) $(PROFILER_DEFINES) -c -o $@ $<
