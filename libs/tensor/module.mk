TENSOR_SRCS = \
	libs/tensor/tensor_impl_wrapper.cpp \
	libs/tensor/tensor_impl.cpp \
	libs/tensor/tensor.cpp \
	libs/tensor/types.cpp \
	libs/tensor/tensor_utils.cpp \
	libs/tensor/serialization.cpp \

TENSOR_LIB = $(LIBDIR)/libtensor.a
TENSOR_DEFINES =
TENSOR_INCLUDES = $(LIBS_INCLUDES)
TENSOR_LDFLAGS = -lcommon -lllrt -ltt_metal
TENSOR_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TENSOR_OBJS = $(addprefix $(OBJDIR)/, $(TENSOR_SRCS:.cpp=.o))
TENSOR_DEPS = $(addprefix $(OBJDIR)/, $(TENSOR_SRCS:.cpp=.d))

-include $(TENSOR_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
libs/tensor: $(TENSOR_LIB)

$(TENSOR_LIB): $(COMMON_LIB) $(TT_METAL_LIB) $(TENSOR_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $(TENSOR_OBJS)

$(OBJDIR)/libs/tensor/%.o: libs/tensor/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TENSOR_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TENSOR_INCLUDES) $(TENSOR_DEFINES) -c -o $@ $<
