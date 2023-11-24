TRACY_INCLUDES = -I$(TT_METAL_HOME)/tt_metal/third_party/tracy/public/tracy/
TRACY_DEFINES = -DTRACY_NO_CONTEXT_SWITCH

#TRACY_DEFINES = -DTRACY_SAMPLING_HZ=40000 -DTRACY_NO_SYSTEM_TRACING  -DTRACY_NO_CALLSTACK -DTRACY_NO_CALLSTACK_INLINES
TRACY_SRCS = \
	tt_metal/third_party/tracy/public/TracyClient.cpp

TRACY_OBJS = $(addprefix $(OBJDIR)/, $(TRACY_SRCS:.cpp=.o))
TRACY_DEPS = $(addprefix $(OBJDIR)/, $(TRACY_SRCS:.cpp=.d))

-include $(TRACY_DEPS)

tracy: $(TRACY_OBJS)

$(OBJDIR)/tt_metal/third_party/tracy/public/%.o: tt_metal/third_party/tracy/public/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TRACY_INCLUDES) $(TRACY_DEFINES) -c -o $@ $<
