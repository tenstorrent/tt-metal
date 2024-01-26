QUEUE_SRCS = \
	tt_eager/queue/queue.cpp \

QUEUE_LIB = $(LIBDIR)/libqueue.a
QUEUE_DEFINES =
QUEUE_INCLUDES = $(TT_EAGER_INCLUDES)
QUEUE_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

QUEUE_OBJS = $(addprefix $(OBJDIR)/, $(QUEUE_SRCS:.cpp=.o))
QUEUE_DEPS = $(addprefix $(OBJDIR)/, $(QUEUE_SRCS:.cpp=.d))

-include $(QUEUE_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_eager/queue: $(QUEUE_LIB)

$(QUEUE_LIB): $(COMMON_LIB) $(TT_METAL_LIB) $(QUEUE_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $(QUEUE_OBJS)

$(OBJDIR)/tt_eager/queue/%.o: tt_eager/queue/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(QUEUE_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(QUEUE_INCLUDES) $(QUEUE_DEFINES) -c -o $@ $<
