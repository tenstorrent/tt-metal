# Every variable in subdir must be prefixed with subdir (emulating a namespace)

LL_BUDA_LIB = $(LIBDIR)/libll_buda.a
LL_BUDA_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
LL_BUDA_INCLUDES = $(COMMON_INCLUDES) -I$(BUDA_HOME)/ll_buda -I$(BUDA_HOME)/.
LL_BUDA_LDFLAGS = -L$(BUDA_HOME) -lcommon -lbuild_kernels_for_riscv -lllrt -lll_buda_impl
LL_BUDA_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

include ll_buda/impl/module.mk

LL_BUDA_SRCS = \
	ll_buda/op_library/eltwise_binary/eltwise_binary_op.cpp \
	ll_buda/op_library/eltwise_unary/eltwise_unary_op.cpp \
	ll_buda/op_library/pad_h_rm/pad_h_rm_op.cpp \
	ll_buda/op_library/transpose/transpose_op.cpp \
	ll_buda/op_library/transpose_rm/transpose_rm_op.cpp \
	ll_buda/op_library/reduce/reduce_op.cpp \
	ll_buda/op_library/bcast/bcast_op.cpp \
	ll_buda/op_library/bmm/bmm_op.cpp \
	ll_buda/op_library/tilize/tilize_op.cpp \
	ll_buda/op_library/untilize/untilize_op.cpp \
	ll_buda/op_library/reshape/reshape_op.cpp \
	ll_buda/tensor/tensor.cpp \
	ll_buda/ll_buda.cpp \

LL_BUDA_OBJS = $(addprefix $(OBJDIR)/, $(LL_BUDA_SRCS:.cpp=.o))
LL_BUDA_DEPS = $(addprefix $(OBJDIR)/, $(LL_BUDA_SRCS:.cpp=.d))

-include $(LL_BUDA_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
ll_buda: $(LL_BUDA_LIB)

$(LL_BUDA_LIB): $(COMMON_LIB) $(LL_BUDA_OBJS) $(LL_BUDA_IMPL_LIB) $(LLRT_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(LL_BUDA_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(LL_BUDA_LDFLAGS)

$(OBJDIR)/ll_buda/%.o: ll_buda/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(LL_BUDA_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(LL_BUDA_INCLUDES) $(LL_BUDA_DEFINES) -c -o $@ $<
