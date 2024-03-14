TT_LIB_LIB = $(LIBDIR)/libtt_lib_csrc.so
TT_LIB_LIB_LOCAL_SO = tt_eager/tt_lib/_C.so
TT_LIB_DEFINES =
TT_LIB_INCLUDES = $(TT_EAGER_INCLUDES) $(shell python3-config --includes) -Itt_metal/third_party/pybind11/include
TT_LIB_LDFLAGS = -ltt_dnn -ldtx -ltensor -lqueue -ltt_metal -lyaml-cpp $(shell python3-config --ldflags --embed) $(LDFLAGS)
TT_LIB_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast  -fno-var-tracking

TT_LIB_SRCS = \
	tt_eager/tt_lib/csrc/tt_lib_bindings.cpp \
	tt_eager/tt_lib/csrc/type_caster.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor_composite_ops.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor_backward_ops.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor_pytensor.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor_dm_ops.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor_custom_bmm_ops.cpp \
	tt_eager/tt_lib/csrc/tt_lib_bindings_tensor_xary_ops.cpp \

TT_LIB_OBJS = $(addprefix $(OBJDIR)/, $(TT_LIB_SRCS:.cpp=.o))
TT_LIB_DEPS = $(addprefix $(OBJDIR)/, $(TT_LIB_SRCS:.cpp=.d))

-include $(TT_LIB_DEPS)

tt_lib: $(TT_LIB_LIB)

# Link obj files into shared lib
$(TT_LIB_LIB): $(TT_LIB_OBJS) $(TT_DNN_LIB) $(TENSOR_LIB) $(DTX_LIB) $(TT_METAL_LIB) $(QUEUE_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_LIB_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $(TT_LIB_OBJS) $(TT_LIB_LDFLAGS)

$(TT_LIB_LIB_LOCAL_SO): $(TT_LIB_LIB)
	cp -fp $^ $@

# Compile obj files
$(OBJDIR)/tt_eager/tt_lib/csrc/%.o: tt_eager/tt_lib/csrc/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_LIB_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_LIB_INCLUDES) -c -o $@ $<

tt_lib/csrc: $(TT_LIB_LIB)
