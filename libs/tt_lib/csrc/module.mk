TT_LIB_LIB = $(LIBDIR)/libtt_lib_csrc.so
TT_LIB_DEFINES =
TT_LIB_INCLUDES = $(LIBS_INCLUDES) $(shell python3-config --includes) -Itt_metal/third_party/pybind11/include
TT_LIB_LDFLAGS = $(LDFLAGS) -ltt_dnn -ldtx -ltensor -ltt_metal -lyaml-cpp -ltracy
TT_LIB_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_LIB_SRCS = \
	libs/tt_lib/csrc/tt_lib_bindings.cpp \
	libs/tt_lib/csrc/type_caster.cpp \

TT_LIB_OBJS = $(addprefix $(OBJDIR)/, $(TT_LIB_SRCS:.cpp=.o))
TT_LIB_DEPS = $(addprefix $(OBJDIR)/, $(TT_LIB_SRCS:.cpp=.d))

-include $(TT_LIB_DEPS)

tt_lib: $(TT_LIB_LIB)

# Link obj files into shared lib
$(TT_LIB_LIB): $(TT_LIB_OBJS) $(TT_DNN_LIB) $(TENSOR_LIB) $(DTX_LIB) $(TT_METAL_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TT_LIB_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $(TT_LIB_OBJS) $(TT_LIB_LDFLAGS)

.PHONY: tt_lib/csrc/setup_local_so
tt_lib/csrc/setup_local_so: $(TT_LIB_LIB)
	rm -f libs/tt_lib/_C.so
	cp $^ libs/tt_lib/_C.so

# Compile obj files
$(OBJDIR)/libs/tt_lib/csrc/%.o: libs/tt_lib/csrc/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_LIB_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_LIB_INCLUDES) -c -o $@ $<

tt_lib/csrc: $(TT_LIB_LIB)
