GPAI_LIB = $(LIBDIR)/libgpai_csrc.so
GPAI_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
GPAI_INCLUDES = $(COMMON_INCLUDES) -I$(BUDA_HOME)/ll_buda -I$(BUDA_HOME)/. $(shell python3-config --includes) -I$(BUDA_HOME)/third_party/pybind11/include
GPAI_LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L$(BUDA_HOME) -ldevice -lcommon -lbuild_kernels_for_riscv -lllrt -lll_buda -lyaml-cpp -lhlkc_api -lprofiler -lbuild_kernels_for_riscv
GPAI_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

GPAI_SRCS = \
	gpai/csrc/gpai_bindings.cpp

GPAI_OBJS = $(addprefix $(OBJDIR)/, $(GPAI_SRCS:.cpp=.o))
GPAI_DEPS = $(addprefix $(OBJDIR)/, $(GPAI_SRCS:.cpp=.d))

-include $(GPAI_DEPS)

gpai: $(GPAI_LIB)

# Link obj files into shared lib
$(GPAI_LIB): $(COMMON_LIB) $(GPAI_OBJS) $(DEVICE_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(GPAI_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -o $@ $^ $(LDFLAGS) $(GPAI_LDFLAGS)

.PHONY: gpai/csrc/setup_inplace_link
gpai/csrc/setup_inplace_link: $(GPAI_LIB)
	@mkdir -p $(PYTHON_ENV)/lib/python3.8/site-packages/gpai
	cp $^ $(PYTHON_ENV)/lib/python3.8/site-packages/gpai/_C.so
	rm -f gpai/gpai/_C.so
	ln -s $(PYTHON_ENV)/lib/python3.8/site-packages/gpai/_C.so gpai/gpai

# Compile obj files
$(OBJDIR)/gpai/csrc/%.o: gpai/csrc/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(GPAI_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(GPAI_INCLUDES) -c -o $@ $<

gpai/csrc: $(GPAI_LIB)
