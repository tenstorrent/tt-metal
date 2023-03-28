# Every variable in subdir must be prefixed with subdir (emulating a namespace)
DTX_LIB = $(LIBDIR)/libdtx.a
DTX_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
DTX_INCLUDES = $(LIBS_INCLUDES) -I$(TT_METAL_HOME)/tt_metal/impl
DTX_LDFLAGS = -L$(TT_METAL_HOME) -lcommon -lllrt -ltt_metal
DTX_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

DTX_SRCS = \
	libs/dtx/dtx.cpp \
	libs/dtx/parallelize_generic_tensor_slice.cpp \
	libs/dtx/pass_collapse_transformations.cpp \
	libs/dtx/pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1.cpp \
	libs/dtx/pass_generate_transfer_addresses.cpp \
	libs/dtx/pass_reverse_transformations.cpp \
	libs/dtx/pass_transpose_xy.cpp \
	libs/dtx/pass_transpose_yz.cpp \
	libs/dtx/pass_row_major_memory_store.cpp \
	libs/dtx/pass_convert_abstract_tensor_to_channels_last_layout.cpp \
	libs/dtx/pass_tilize.cpp \
	libs/dtx/pass_util.cpp \
	libs/dtx/util.cpp \
	libs/dtx/util_vector_of_ints.cpp

DTX_OBJS = $(addprefix $(OBJDIR)/, $(DTX_SRCS:.cpp=.o))
DTX_DEPS = $(addprefix $(OBJDIR)/, $(DTX_SRCS:.cpp=.d))

-include $(DTX_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
libs/dtx: $(DTX_LIB)

$(DTX_LIB): $(COMMON_LIB) $(TT_METAL_LIB) $(DTX_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(DTX_OBJS)

$(OBJDIR)/libs/dtx/%.o: libs/dtx/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(DTX_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(DTX_INCLUDES) $(DTX_DEFINES) -c -o $@ $<
