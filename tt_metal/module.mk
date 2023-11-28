# Every variable in subdir must be prefixed with subdir (emulating a namespace)

CFLAGS += -DFMT_HEADER_ONLY -I$(TT_METAL_HOME)/tt_metal/third_party/fmt

include $(TT_METAL_HOME)/tt_metal/common/module.mk
include $(TT_METAL_HOME)/tt_metal/build_kernels_for_riscv/module.mk
include $(TT_METAL_HOME)/tt_metal/llrt/module.mk
include $(TT_METAL_HOME)/tt_metal/tools/module.mk

ifeq ($(TT_METAL_CREATE_STATIC_LIB), 1)
TT_METAL_LIB = $(LIBDIR)/libtt_metal.a
else
TT_METAL_LIB = $(LIBDIR)/libtt_metal.so
endif
TT_METAL_LDFLAGS = $(LDFLAGS) -ltt_metal_detail -ltt_metal_impl -lprofiler -lllrt -ldevice -lbuild_kernels_for_riscv -lcommon
TT_METAL_INCLUDES = $(COMMON_INCLUDES)
TT_METAL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

include $(TT_METAL_HOME)/tt_metal/impl/module.mk
include $(TT_METAL_HOME)/tt_metal/detail/module.mk

TT_METAL_SRCS = \
	tt_metal/tt_metal.cpp

TT_METAL_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_SRCS:.cpp=.o))
TT_METAL_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_SRCS:.cpp=.d))

-include $(TT_METAL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal: $(TT_METAL_LIB) tools

ifeq ($(TT_METAL_CREATE_STATIC_LIB), 1)
# If production build, release all of tt_metal as a full static library for later build with Eager wheel
$(TT_METAL_LIB): $(COMMON_OBJS) $(TT_METAL_OBJS) $(DEVICE_OBJS) $(TT_METAL_IMPL_OBJS) $(TT_METAL_DETAIL_OBJS) $(LLRT_OBJS) $(BUILD_KERNELS_FOR_RISCV_OBJS) $(PROFILER_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $^
else
$(TT_METAL_LIB): $(TT_METAL_OBJS) $(COMMON_LIB) $(DEVICE_LIB) $(TT_METAL_IMPL_LIB) $(TT_METAL_DETAIL_LIB) $(LLRT_LIB) $(BUILD_KERNELS_FOR_RISCV_LIB) $(PROFILER_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(TT_METAL_LDFLAGS)
endif

# TODO: rk: need to use a general way to do the following directives, note that using tt_metal/%.o will
# include EVERYTHING under tt_metal, forcing the build step to use only build directives in this file
# rather than the specialized ones in each submodule
$(OBJDIR)/tt_metal/tt_metal.o: tt_metal/tt_metal.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) -c -o $@ $<
