# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TENSOR_LIB = $(LIBDIR)/libtensor.a
TENSOR_SRCS = \
	tensor/tensor.cpp

TENSOR_INCLUDES = -I$(TT_METAL_HOME)/tensor $(COMMON_INCLUDES)

TENSOR_OBJS = $(addprefix $(OBJDIR)/, $(TENSOR_SRCS:.cpp=.o))
TENSOR_DEPS = $(addprefix $(OBJDIR)/, $(TENSOR_SRCS:.cpp=.d))

-include $(TENSOR_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tensor: $(TENSOR_LIB)

# TODO: add src/firmware src/ckernel when they become real targets
$(TENSOR_LIB): $(COMMON_LIB) $(TENSOR_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(TENSOR_OBJS)

.PRECIOUS: $(OBJDIR)/tensor/%.o
$(OBJDIR)/tensor/%.o: tensor/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TENSOR_INCLUDES) -c -o $@ $<
