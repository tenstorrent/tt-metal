# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_METAL_LIB = $(LIBDIR)/libtt_metal.a
TT_METAL_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_METAL_INCLUDES = $(COMMON_INCLUDES) -I$(BUDA_HOME)/tt_metal -I$(BUDA_HOME)/.
TT_METAL_LDFLAGS = -L$(BUDA_HOME) -lcommon -lbuild_kernels_for_riscv -lllrt -ltt_metal_impl
TT_METAL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

include tt_metal/impl/module.mk

TT_METAL_SRCS = \
	tt_metal/op_library/eltwise_binary/eltwise_binary_op.cpp \
	tt_metal/op_library/eltwise_unary/eltwise_unary_op.cpp \
	tt_metal/op_library/eltwise_unary/single_core/eltwise_unary_op_single_core.cpp \
	tt_metal/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.cpp \
	tt_metal/op_library/pad_h_rm/pad_h_rm_op.cpp \
	tt_metal/op_library/transpose/transpose_op.cpp \
	tt_metal/op_library/transpose_rm/transpose_rm_op.cpp \
	tt_metal/op_library/reduce/reduce_op.cpp \
	tt_metal/op_library/bcast/bcast_op.cpp \
	tt_metal/op_library/bmm/bmm_op.cpp \
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

$(TT_METAL_LIB): $(COMMON_LIB) $(TT_METAL_OBJS) $(TT_METAL_IMPL_LIB) $(LLRT_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(TT_METAL_LDFLAGS)

$(OBJDIR)/tt_metal/%.o: tt_metal/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_INCLUDES) $(TT_METAL_DEFINES) -c -o $@ $<
