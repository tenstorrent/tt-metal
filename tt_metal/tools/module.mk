# Every variable in subdir must be prefixed with subdir (emulating a namespace)

# include $(TT_METAL_HOME)/tt_metal/tools/tt_gdb/module.mk # needs to compiled after llrt and tt_metal
include $(TT_METAL_HOME)/tt_metal/tools/profiler/module.mk
include $(TT_METAL_HOME)/tt_metal/tools/watcher_dump/module.mk

TOOLS = \
	tools/memset

TOOLS_SRCS = $(addprefix tt_metal/, $(addsuffix .cpp, $(TOOLS)))

TOOLS_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tools
TOOLS_LDFLAGS = $(LDFLAGS) -lyaml-cpp -lstdc++fs -lhwloc -lm

TOOLS_DEPS = $(addprefix $(OBJDIR)/, $(TOOLS_SRCS:.cpp=.d))

-include $(TOOLS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tools: $(OBJDIR)/tt_metal/tools/memset tools/profiler tools/watcher_dump #tools/tt_gdb

.PRECIOUS: $(OBJDIR)/tools/%
$(OBJDIR)/tt_metal/tools/memset: $(OBJDIR)/tt_metal/tools/memset.o $(COMMON_OBJS) $(LLRT_OBJS) $(DEVICE_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TOOLS_INCLUDES) -o $@ $^ $(TOOLS_LDFLAGS)

$(OBJDIR)/tt_metal/tools/memset.o: tt_metal/tools/memset.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TOOLS_INCLUDES) -c -o $@ $<
