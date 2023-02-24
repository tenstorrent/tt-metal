# Every variable in subdir must be prefixed with subdir (emulating a namespace)
LL_BUDA_IMPL_LIB = $(LIBDIR)/libll_buda_impl.a
LL_BUDA_IMPL_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
LL_BUDA_IMPL_INCLUDES = $(COMMON_INCLUDES) $(MODEL_INCLUDES) -I$(BUDA_HOME)/ll_buda/impl -I$(BUDA_HOME)/.
LL_BUDA_IMPL_LDFLAGS = -L$(BUDA_HOME) -lcommon -lllrt
LL_BUDA_IMPL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

LL_BUDA_IMPL_SRCS = \
	ll_buda/impl/device/device.cpp \
	ll_buda/impl/device/memory_manager.cpp \
	ll_buda/impl/buffers/buffer.cpp \
	ll_buda/impl/buffers/interleaved_buffer.cpp \
	ll_buda/impl/kernels/kernel_args.cpp \
	ll_buda/impl/kernels/kernel.cpp \
	ll_buda/impl/program.cpp \
	ll_buda/impl/dtx/dtx.cpp \
	ll_buda/impl/dtx/parallelize_generic_tensor_slice.cpp \
	ll_buda/impl/dtx/pass_collapse_transformations.cpp \
	ll_buda/impl/dtx/pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1.cpp \
	ll_buda/impl/dtx/pass_generate_transfer_addresses.cpp \
	ll_buda/impl/dtx/pass_reverse_transformations.cpp \
	ll_buda/impl/dtx/pass_tilize.cpp \
	ll_buda/impl/dtx/pass_util.cpp \
	ll_buda/impl/dtx/util.cpp \
	ll_buda/impl/dtx/util_vector_of_ints.cpp \
	ll_buda/impl/dtx/evaluate.cpp \
	common/utils.cpp \

LL_BUDA_IMPL_OBJS = $(addprefix $(OBJDIR)/, $(LL_BUDA_IMPL_SRCS:.cpp=.o))
LL_BUDA_IMPL_DEPS = $(addprefix $(OBJDIR)/, $(LL_BUDA_IMPL_SRCS:.cpp=.d))

-include $(LL_BUDA_IMPL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
ll_buda/impl: $(LL_BUDA_IMPL_LIB)

$(LL_BUDA_IMPL_LIB): $(COMMON_LIB) $(LL_BUDA_IMPL_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(LL_BUDA_IMPL_OBJS)

$(OBJDIR)/ll_buda/impl/%.o: ll_buda/impl/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(LL_BUDA_IMPL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(LL_BUDA_IMPL_INCLUDES) $(LL_BUDA_IMPL_DEFINES) -c -o $@ $<
