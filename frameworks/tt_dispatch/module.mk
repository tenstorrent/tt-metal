TT_DISPATCH_SRCS = \
	frameworks/tt_dispatch/impl/command_queue_interface.cpp \
	frameworks/tt_dispatch/impl/command_queue.cpp \
	frameworks/tt_dispatch/impl/command.cpp \

TT_DISPATCH_LIB = $(LIBDIR)/libtt_dispatch.a
TT_DISPATCH_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_DISPATCH_INCLUDES = $(LIBS_INCLUDES)
TT_DISPATCH_LDFLAGS = -L$(TT_METAL_HOME) -lcommon -lllrt -ltt_metal
TT_DISPATCH_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_DISPATCH_OBJS = $(addprefix $(OBJDIR)/, $(TT_DISPATCH_SRCS:.cpp=.o))
TT_DISPATCH_DEPS = $(addprefix $(OBJDIR)/, $(TT_DISPATCH_SRCS:.cpp=.d))

-include $(TT_DISPATCH_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
frameworks/tt_dispatch: $(TT_DISPATCH_LIB)

$(TT_DISPATCH_LIB): $(COMMON_LIB) $(TT_METAL_LIB) $(TT_DISPATCH_OBJS)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_DISPATCH_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(TT_DISPATCH_LDFLAGS)

$(OBJDIR)/frameworks/tt_dispatch/%.o: frameworks/tt_dispatch/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_DISPATCH_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_DISPATCH_INCLUDES) $(TT_DISPATCH_DEFINES) -c -o $@ $<
