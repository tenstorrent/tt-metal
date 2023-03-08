# Every variable in subdir must be prefixed with subdir (emulating a namespace)

LLRT_LIB = $(LIBDIR)/libllrt.a
LLRT_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
LLRT_INCLUDES = $(COMMON_INCLUDES) $(MODEL_INCLUDES) $(NETLIST_INCLUDES) -I$(TT_METAL_HOME)/llrt -I$(TT_METAL_HOME)/.
LLRT_LDFLAGS = -L$(TT_METAL_HOME) -ldevice -lcommon
LLRT_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

LLRT_SRCS = \
	llrt/tt_cluster.cpp \
	llrt/tt_debug_print_server.cpp \
	llrt/llrt.cpp

LLRT_OBJS = $(addprefix $(OBJDIR)/, $(LLRT_SRCS:.cpp=.o))
LLRT_DEPS = $(addprefix $(OBJDIR)/, $(LLRT_SRCS:.cpp=.d))

-include $(LLRT_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
llrt: $(LLRT_LIB)

$(LLRT_LIB): $(COMMON_LIB) $(NETLIST_LIB) $(LLRT_OBJS) $(DEVICE_LIB)
	@mkdir -p $(@D)
	ar rcs -o $@ $(LLRT_OBJS)

$(OBJDIR)/llrt/%.o: llrt/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(LLRT_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(LLRT_INCLUDES) $(LLRT_DEFINES) -c -o $@ $<
