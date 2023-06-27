# Every variable in subdir must be prefixed with subdir (emulating a namespace)

CFLAGS += -DFMT_HEADER_ONLY -I$(TT_METAL_HOME)/tt_metal/third_party/fmt

include $(TT_METAL_HOME)/tt_metal/common/module.mk
include $(TT_METAL_HOME)/tt_metal/device/module.mk
include $(TT_METAL_HOME)/tt_metal/src/firmware/module.mk
include $(TT_METAL_HOME)/tt_metal/src/ckernels/module.mk
include $(TT_METAL_HOME)/tt_metal/tools/module.mk
include $(TT_METAL_HOME)/tt_metal/build_kernels_for_riscv/module.mk
include $(TT_METAL_HOME)/tt_metal/llrt/module.mk
include $(TT_METAL_HOME)/tt_metal/python_env/module.mk

# Programming examples for external users
include $(TT_METAL_HOME)/tt_metal/programming_examples/module.mk

TT_METAL_LIB = $(LIBDIR)/libtt_metal.a
TT_METAL_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_METAL_INCLUDES = $(COMMON_INCLUDES)
TT_METAL_LDFLAGS = -L$(TT_METAL_HOME) -lcommon -lbuild_kernels_for_riscv -lllrt -ltt_metal_impl -ltt_metal_detail
TT_METAL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

include $(TT_METAL_HOME)/tt_metal/impl/module.mk
include $(TT_METAL_HOME)/tt_metal/detail/module.mk

TT_METAL_SRCS = \
	tt_metal/tt_metal.cpp

TT_METAL_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_SRCS:.cpp=.o))
TT_METAL_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_SRCS:.cpp=.d))

-include $(TT_METAL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal: $(TT_METAL_LIB)

$(TT_METAL_LIB): $(COMMON_LIB) $(TT_METAL_OBJS) $(TT_METAL_IMPL_LIB) $(TT_METAL_DETAIL_LIB) $(LLRT_LIB) $(BUILD_KERNELS_FOR_RISCV_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(TT_METAL_LDFLAGS)

# TODO: rk: need to use a general way to do the following directives, note that using tt_metal/%.o will
# include EVERYTHING under tt_metal, forcing the build step to use only build directives in this file
# rather than the specialized ones in each submodule
$(OBJDIR)/tt_metal/tt_metal.o: tt_metal/tt_metal.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) $(TT_METAL_DEFINES) -c -o $@ $<
