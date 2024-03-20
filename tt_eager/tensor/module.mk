TENSOR_SRCS = \
	tt_eager/tensor/tensor_impl_wrapper.cpp \
	tt_eager/tensor/tensor_impl.cpp \
	tt_eager/tensor/tensor.cpp \
	tt_eager/tensor/types.cpp \
	tt_eager/tensor/tensor_utils.cpp \
	tt_eager/tensor/serialization.cpp \
	tt_eager/tensor/operation_history.cpp \

TENSOR_LIB = $(LIBDIR)/libtensor.a
TENSOR_DEFINES =
TENSOR_INCLUDES = $(TT_EAGER_INCLUDES)
TENSOR_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TENSOR_OBJS = $(addprefix $(OBJDIR)/, $(TENSOR_SRCS:.cpp=.o))
TENSOR_DEPS = $(addprefix $(OBJDIR)/, $(TENSOR_SRCS:.cpp=.d))

-include $(TENSOR_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_eager/tensor: $(TENSOR_LIB)

$(TENSOR_LIB): $(COMMON_LIB) $(TT_METAL_LIB) $(TENSOR_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $(TENSOR_OBJS)

$(OBJDIR)/tt_eager/tensor/%.o: tt_eager/tensor/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TENSOR_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TENSOR_INCLUDES) $(TENSOR_DEFINES) -c -o $@ $<
