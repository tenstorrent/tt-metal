TTLIB_LIB = $(LIBDIR)/libttlib_csrc.so
TTLIB_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TTLIB_INCLUDES = $(COMMON_INCLUDES) $(shell python3-config --includes) -I$(TT_METAL_HOME)/tt_metal/third_party/pybind11/include
TTLIB_LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L$(TT_METAL_HOME) -ldevice -lcommon -lbuild_kernels_for_riscv -lllrt -ltt_metal -ltt_dnn -lyaml-cpp -lprofiler -lbuild_kernels_for_riscv
TTLIB_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TTLIB_SRCS = \
	libs/ttlib/csrc/ttlib_bindings.cpp \
	libs/ttlib/csrc/type_caster.cpp \

TTLIB_OBJS = $(addprefix $(OBJDIR)/, $(TTLIB_SRCS:.cpp=.o))
TTLIB_DEPS = $(addprefix $(OBJDIR)/, $(TTLIB_SRCS:.cpp=.d))

-include $(TTLIB_DEPS)

ttlib: $(TTLIB_LIB)

# Link obj files into shared lib
$(TTLIB_LIB): $(COMMON_LIB) $(TTLIB_OBJS) $(DEVICE_LIB) $(TT_DNN_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(TTLIB_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(TTLIB_LDFLAGS)

.PHONY: ttlib/csrc/setup_local_so
ttlib/csrc/setup_local_so: $(TTLIB_LIB)
	rm -f $(TT_METAL_HOME)/libs/ttlib/_C.so
	cp $^ $(TT_METAL_HOME)/libs/ttlib/_C.so

# Compile obj files
$(OBJDIR)/libs/ttlib/csrc/%.o: libs/ttlib/csrc/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TTLIB_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TTLIB_INCLUDES) -c -o $@ $<

ttlib/csrc: $(TTLIB_LIB)
