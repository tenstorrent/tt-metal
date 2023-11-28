# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_DETAIL_LIB = $(LIBDIR)/libtt_metal_detail.a
TT_METAL_DETAIL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_METAL_DETAIL_SRCS = \
	tt_metal/detail/reports/compilation_reporter.cpp \
	tt_metal/detail/reports/memory_reporter.cpp \

TT_METAL_DETAIL_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_DETAIL_SRCS:.cpp=.o))
TT_METAL_DETAIL_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_DETAIL_SRCS:.cpp=.d))

-include $(TT_METAL_DETAIL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal/detail: $(TT_METAL_DETAIL_LIB)

$(TT_METAL_DETAIL_LIB): $(COMMON_LIB) $(TT_METAL_IMPL_LIB) $(TT_METAL_DETAIL_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(TT_METAL_DETAIL_OBJS)

$(OBJDIR)/tt_metal/detail/%.o: tt_metal/detail/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_DETAIL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) $(TT_METAL_DEFINES) -c -o $@ $<
