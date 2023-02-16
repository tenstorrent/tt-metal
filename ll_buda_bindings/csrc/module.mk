LL_BUDA_BINDINGS_LIB = $(LIBDIR)/libll_buda_csrc.so
LL_BUDA_BINDINGS_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
LL_BUDA_BINDINGS_INCLUDES = $(COMMON_INCLUDES) -I$(BUDA_HOME)/ll_buda -I$(BUDA_HOME)/. $(shell python3-config --includes) -I$(BUDA_HOME)/third_party/pybind11/include 
LL_BUDA_BINDINGS_LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L$(BUDA_HOME) -ldevice -lcommon -lbuild_kernels_for_riscv -lllrt -lll_buda -lyaml-cpp -lhlkc_api -lprofiler -lbuild_kernels_for_riscv
LL_BUDA_BINDINGS_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

LL_BUDA_BINDINGS_SRCS = \
	ll_buda_bindings/csrc/ll_buda_bindings.cpp

LL_BUDA_BINDINGS_OBJS = $(addprefix $(OBJDIR)/, $(LL_BUDA_BINDINGS_SRCS:.cpp=.o))
LL_BUDA_BINDINGS_DEPS = $(addprefix $(OBJDIR)/, $(LL_BUDA_BINDINGS_SRCS:.cpp=.d))

-include $(LL_BUDA_BINDINGS_DEPS)

ll_buda_bindings: $(LL_BUDA_BINDINGS_LIB)

# Link obj files into shared lib
$(LL_BUDA_BINDINGS_LIB): $(COMMON_LIB) $(LL_BUDA_BINDINGS_OBJS) $(DEVICE_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(LL_BUDA_BINDINGS_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(LL_BUDA_BINDINGS_LDFLAGS)

.PHONY: ll_buda_bindings/csrc/setup_inplace_link
ll_buda_bindings/csrc/setup_inplace_link: $(LL_BUDA_BINDINGS_LIB)
	@mkdir -p $(PYTHON_ENV)/lib/python3.8/site-packages/ll_buda_bindings
	cp $^ $(PYTHON_ENV)/lib/python3.8/site-packages/ll_buda_bindings/_C.so
	rm -f ll_buda_bindings/ll_buda_bindings/_C.so
	ln -s $(PYTHON_ENV)/lib/python3.8/site-packages/ll_buda_bindings/_C.so ll_buda_bindings/ll_buda_bindings

# Compile obj files
$(OBJDIR)/ll_buda_bindings/csrc/%.o: ll_buda_bindings/csrc/%.cpp 
	@mkdir -p $(@D)
	$(CXX) $(LL_BUDA_BINDINGS_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(LL_BUDA_BINDINGS_INCLUDES) -c -o $@ $<

ll_buda_bindings/csrc: $(LL_BUDA_BINDINGS_LIB)
