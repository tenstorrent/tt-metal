# Every variable in subdir must be prefixed with subdir (emulating a namespace)
LLRT_DEFINES =
LLRT_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tt_metal/llrt
LLRT_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

LLRT_SRCS_RELATIVE = \
	llrt/tt_cluster.cpp \
	llrt/llrt.cpp \
	llrt/rtoptions.cpp \
	llrt/tt_memory.cpp \
	llrt/tt_hexfile.cpp

LLRT_SRCS = $(addprefix tt_metal/, $(LLRT_SRCS_RELATIVE))

LLRT_OBJS = $(addprefix $(OBJDIR)/, $(LLRT_SRCS:.cpp=.o))
LLRT_DEPS = $(addprefix $(OBJDIR)/, $(LLRT_SRCS:.cpp=.d))

-include $(LLRT_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
llrt: $(LLRT_OBJS) $(DEVICE_OBJS) $(COMMON_OBJS)

$(OBJDIR)/tt_metal/llrt/%.o: tt_metal/llrt/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(LLRT_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(LLRT_INCLUDES) $(LLRT_DEFINES) -c -o $@ $<
