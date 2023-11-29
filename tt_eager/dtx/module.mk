# Every variable in subdir must be prefixed with subdir (emulating a namespace)
DTX_DEFINES =
DTX_INCLUDES = $(TT_EAGER_INCLUDES)
DTX_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

DTX_SRCS = \
	tt_eager/dtx/dtx.cpp \
	tt_eager/dtx/dtx_evaluate.cpp \
	tt_eager/dtx/parallelize_generic_tensor_slice.cpp \
	tt_eager/dtx/pass_collapse_transformations.cpp \
	tt_eager/dtx/pass_convert_tensor_layout_3d_conv_act_to_2Dmatrix.cpp \
	tt_eager/dtx/pass_generate_transfer_addresses.cpp \
	tt_eager/dtx/pass_reverse_transformations.cpp \
	tt_eager/dtx/pass_transpose_xy.cpp \
	tt_eager/dtx/pass_transpose_yz.cpp \
	tt_eager/dtx/pass_row_major_memory_store.cpp \
	tt_eager/dtx/pass_row_major_memory_store_blocks.cpp \
	tt_eager/dtx/pass_convert_abstract_tensor_to_channels_last_layout.cpp \
	tt_eager/dtx/pass_tilize.cpp \
	tt_eager/dtx/pass_pad_2d_matrix.cpp \
	tt_eager/dtx/pass_block_2d_matrix.cpp \
	tt_eager/dtx/pass_block_2d_with_duplicate_blocks.cpp \
	tt_eager/dtx/pass_generate_groups_outermost_dim.cpp \
	tt_eager/dtx/pass_util.cpp \
	tt_eager/dtx/pass_simple_high_level_pass.cpp \
	tt_eager/dtx/pass_conv_transform.cpp \
	tt_eager/dtx/util.cpp \
	tt_eager/dtx/util_vector_of_ints.cpp

DTX_OBJS = $(addprefix $(OBJDIR)/, $(DTX_SRCS:.cpp=.o))
DTX_DEPS = $(addprefix $(OBJDIR)/, $(DTX_SRCS:.cpp=.d))

-include $(DTX_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_eager/dtx: $(DTX_OBJS) $(TENSOR_OBJS)

$(OBJDIR)/tt_eager/dtx/%.o: tt_eager/dtx/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(DTX_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(DTX_INCLUDES) $(DTX_DEFINES) -c -o $@ $<
