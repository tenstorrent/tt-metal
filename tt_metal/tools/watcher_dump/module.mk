# Every variable in subdir must be prefixed with subdir (emulating a namespace)
WATCHER_DUMP_INCLUDES = $(COMMON_INCLUDES) -Itools/watcher_dump
WATCHER_DUMP_DEFINES =
WATCHER_DUMP_LDFLAGS = $(LDFLAGS) -ltt_metal -lyaml-cpp

WATCHER_DUMP_SRCS += \
	$(wildcard tt_metal/tools/watcher_dump/*.cpp)
WATCHER_DUMP_OBJS = $(addprefix $(OBJDIR)/, $(WATCHER_DUMP_SRCS:.cpp=.o))
WATCHER_DUMP_DEPS = $(addprefix $(OBJDIR)/, $(WATCHER_DUMP_SRCS:.cpp=.d))

-include $(WATCHER_DUMP_DEPS)
# Each module has a top level target as the entrypoint which must match the subdir name
tools/watcher_dump: $(WATCHER_DUMP_OBJS) $(OUT)/tools/watcher_dump

$(OBJDIR)/tt_metal/tools/watcher_dump/%.o: tt_metal/tools/watcher_dump/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(WATCHER_DUMP_INCLUDES) $(WATCHER_DUMP_DEFINES) -c -o $@ $<

$(OUT)/tools/watcher_dump: $(WATCHER_DUMP_OBJS) $(OUT)/lib/libtt_metal.so
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(WATCHER_DUMP_INCLUDES) $(WATCHER_DUMP_DEFINES) -o $@ $^ $(WATCHER_DUMP_LDFLAGS)
