# Every variable in subdir must be prefixed with subdir (emulating a namespace)

BASE_INCLUDES+=-I./ -I./tt_metal/ # make all tt_metal modules includable

CFLAGS += -DFMT_HEADER_ONLY -I$(TT_METAL_HOME)/tt_metal/third_party/fmt

include $(TT_METAL_HOME)/tt_metal/common/module.mk
include $(TT_METAL_HOME)/tt_metal/device/module.mk
include $(TT_METAL_HOME)/src/ckernels/module.mk
include $(TT_METAL_HOME)/src/firmware/module.mk
include $(TT_METAL_HOME)/hlkc/module.mk
include $(TT_METAL_HOME)/tt_metal/tools/module.mk
include $(TT_METAL_HOME)/tt_metal/build_kernels_for_riscv/module.mk
include $(TT_METAL_HOME)/tt_metal/llrt/module.mk
include $(TT_METAL_HOME)/tt_metal/python_env/module.mk

# Programming examples for external users
include $(TT_METAL_HOME)/tt_metal/programming_examples/module.mk

TT_METAL_LIB = $(LIBDIR)/libtt_metal.a
TT_METAL_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_METAL_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tt_metal -I$(TT_METAL_HOME)/.
TT_METAL_LDFLAGS = -L$(TT_METAL_HOME) -lcommon -lbuild_kernels_for_riscv -lllrt -ltt_metal_impl
TT_METAL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

include tt_metal/impl/module.mk

TT_METAL_SRCS = \
	tt_metal/op_library/eltwise_binary/eltwise_binary_op.cpp \
	tt_metal/op_library/eltwise_binary/single_core/eltwise_binary_op_single_core.cpp \
	tt_metal/op_library/eltwise_binary/multi_core/eltwise_binary_op_multi_core.cpp \
	tt_metal/op_library/eltwise_unary/eltwise_unary_op.cpp \
	tt_metal/op_library/eltwise_unary/single_core/eltwise_unary_op_single_core.cpp \
	tt_metal/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.cpp \
	tt_metal/op_library/pad_h_rm/pad_h_rm_op.cpp \
	tt_metal/op_library/fill_rm/fill_rm_op.cpp \
	tt_metal/op_library/transpose/transpose_op.cpp \
	tt_metal/op_library/transpose/wh_multi_core/transpose_wh_op_multi_core.cpp \
	tt_metal/op_library/transpose/hc_multi_core/transpose_hc_op_multi_core.cpp \
	tt_metal/op_library/transpose/single_core/transpose_op_single_core.cpp \
	tt_metal/op_library/transpose_rm/transpose_rm_op.cpp \
	tt_metal/op_library/reduce/reduce_op.cpp \
	tt_metal/op_library/reduce/single_core/reduce_op_single_core.cpp \
	tt_metal/op_library/reduce/multi_core_h/reduce_op_multi_core_h.cpp \
	tt_metal/op_library/reduce/multi_core_w/reduce_op_multi_core_w.cpp \
	tt_metal/op_library/reduce/multi_core_hw/reduce_op_multi_core_hw.cpp \
	tt_metal/op_library/bcast/bcast_op.cpp \
	tt_metal/op_library/bcast/single_core/bcast_op_single_core.cpp \
	tt_metal/op_library/bcast/multi_core_h/bcast_op_multi_core_h.cpp \
	tt_metal/op_library/bcast/multi_core_w/bcast_op_multi_core_w.cpp \
	tt_metal/op_library/bcast/multi_core_hw/bcast_op_multi_core_hw.cpp \
	tt_metal/op_library/bmm/bmm_op.cpp \
	tt_metal/op_library/bmm/single_core/bmm_op_single_core.cpp \
	tt_metal/op_library/bmm/multi_core/bmm_op_multi_core.cpp \
	tt_metal/op_library/bmm/multi_core_reuse/bmm_op_multi_core_reuse.cpp \
	tt_metal/op_library/bmm/multi_core_reuse_mcast/bmm_op_multi_core_reuse_mcast.cpp \
	tt_metal/op_library/tilize/tilize_op.cpp \
	tt_metal/op_library/untilize/untilize_op.cpp \
	tt_metal/op_library/reshape/reshape_op.cpp \
	tt_metal/tensor/tensor_impl_wrapper.cpp \
	tt_metal/tensor/tensor_impl.cpp \
	tt_metal/tensor/tensor.cpp \
	tt_metal/tt_metal.cpp \

TT_METAL_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_SRCS:.cpp=.o))
TT_METAL_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_SRCS:.cpp=.d))

-include $(TT_METAL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal: $(TT_METAL_LIB)

$(TT_METAL_LIB): $(COMMON_LIB) $(TT_METAL_OBJS) $(TT_METAL_IMPL_LIB) $(LLRT_LIB) $(BUILD_KERNELS_FOR_RISCV_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(TT_METAL_LDFLAGS)

# TODO: rk: need to use a general way to do the following directives, note that using tt_metal/%.o will
# include EVERYTHING under tt_metal, forcing the build step to use only build directives in this file
# rather than the specialized ones in each submodule
$(OBJDIR)/tt_metal/tt_metal.o: tt_metal/tt_metal.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) $(TT_METAL_DEFINES) -c -o $@ $<

$(OBJDIR)/tt_metal/op_library/%.o: tt_metal/op_library/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) $(TT_METAL_DEFINES) -c -o $@ $<

$(OBJDIR)/tt_metal/tensor/%.o: tt_metal/tensor/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) $(TT_METAL_DEFINES) -c -o $@ $<
