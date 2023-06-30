# Every variable in subdir must be prefixed with subdir (emulating a namespace)
DEVICE_LIB = $(LIBDIR)/libdevice.so
DEVICE_SRCS = \
	tt_metal/device/tt_device.cpp \
	tt_metal/device/tt_memory.cpp \
	tt_metal/device/tt_hexfile.cpp \
	tt_metal/device/tt_silicon_driver.cpp \
	tt_metal/device/tt_silicon_driver_common.cpp

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  DEVICE_SRCS += tt_metal/device/wormhole/impl_device.cpp
else
  DEVICE_SRCS += tt_metal/device/$(ARCH_NAME)/impl_device.cpp
endif

ifeq ($(DISABLE_VERSIM_BUILD),1)
  DEVICE_SRCS += tt_metal/device/tt_versim_stub.cpp
else
  DEVICE_SRCS += tt_metal/device/tt_versim_device.cpp
endif

DEVICE_OBJS = $(addprefix $(OBJDIR)/, $(DEVICE_SRCS:.cpp=.o))
DEVICE_DEPS = $(addprefix $(OBJDIR)/, $(DEVICE_SRCS:.cpp=.d))

DEVICE_INCLUDES=      	\
  -DFMT_HEADER_ONLY     \
  $(COMMON_INCLUDES)    \
  -I$(TT_METAL_HOME)/tt_metal/third_party/fmt

ifeq ($(DISABLE_VERSIM_BUILD),1)
  DEVICE_INCLUDES += -DDISABLE_VERSIM_BUILD
endif

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  DEVICE_INCLUDES += -I$(TT_METAL_HOME)/tt_metal/device/wormhole/
else
  DEVICE_INCLUDES += -I$(TT_METAL_HOME)/tt_metal/device/$(ARCH_NAME)/
endif


ifeq ("$(ARCH_NAME)", "wormhole_b0")
  DEVICE_INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/wormhole
  DEVICE_INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/wormhole/wormhole_b0_defines
else
  DEVICE_INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/grayskull
endif

ifeq ("$(ARCH_NAME)", "wormhole")
  DEVICE_INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/wormhole/wormhole_a0_defines
endif

DEVICE_LDFLAGS = -Wl,-rpath,$(TT_METAL_HOME)/tt_metal/third_party/common_lib
ifdef DEVICE_VERSIM_INSTALL_ROOT
DEVICE_LDFLAGS += -Wl,-rpath,$(DEVICE_VERSIM_INSTALL_ROOT)/versim/$(ARCH_NAME)/lib,-rpath,$(DEVICE_VERSIM_INSTALL_ROOT)/versim/common_lib
endif
DEVICE_LDFLAGS += \
	-L$(TT_METAL_HOME)/tt_metal/third_party/common_lib \
	-lz \
	-l:libboost_system.so.1.65.1 \
	-l:libboost_filesystem.so.1.65.1 \
	-l:libicudata.so.60 \
	-l:libicui18n.so.60 \
	-l:libicuuc.so.60 \
	-l:libboost_thread.so.1.65.1 \
	-l:libboost_regex.so.1.65.1 \
	-lpthread \
	-latomic \
	-lcommon

ifneq ($(DISABLE_VERSIM_BUILD),1)
  DEVICE_LDFLAGS += -l:versim-core.so -l:libc_verilated_hw.so
endif

CLANG_WARNINGS := $(filter-out -Wmaybe-uninitialized,$(WARNINGS))
CLANG_WARNINGS += -Wsometimes-uninitialized
DEVICE_CXX = /usr/bin/clang++-6.0
# TODO: rk: delete both includes here
DEVICE_CXXFLAGS = -MMD $(CLANG_WARNINGS) -I$(TT_METAL_HOME)/. -I$(TT_METAL_HOME)/tt_metal/. --std=c++17
ifeq ($(CONFIG), release)
DEVICE_CXXFLAGS += -O3
else ifeq ($(CONFIG), ci)
DEVICE_CXXFLAGS += -O3  # significantly smaller artifacts
else ifeq ($(CONFIG), assert)
DEVICE_CXXFLAGS += -O3 -g
else ifeq ($(CONFIG), asan)
DEVICE_CXXFLAGS += -O3 -g -fsanitize=address
else ifeq ($(CONFIG), ubsan)
DEVICE_CXXFLAGS += -O3 -g -fsanitize=undefined
else ifeq ($(CONFIG), debug)
DEVICE_CXXFLAGS += -O0 -g -DDEBUG
else
$(error Unknown value for CONFIG "$(CONFIG)")
endif

ifneq ($(filter "$(ARCH_NAME)","wormhole" "wormhole_b0"),)
	DEVICE_CXXFLAGS += -DEN_DRAM_ALIAS
endif

ifeq ($(DISABLE_VERSIM_BUILD),1)
  DEVICE_CXXFLAGS += -DTT_BACKEND_DISABLE_VERSIM_BUILD
endif
ifeq ($(ISSUE_3487_FIX), 1)
  DEVICE_CXXFLAGS += -DISSUE_3487_FIX
endif


-include $(DEVICE_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
device: $(DEVICE_LIB)

$(DEVICE_LIB): $(COMMON_LIB) $(DEVICE_OBJS)
	@mkdir -p $(LIBDIR)
	$(DEVICE_CXX) $(DEVICE_CXXFLAGS) $(SHARED_LIB_FLAGS) -o $(DEVICE_LIB) $^ $(LDFLAGS) $(DEVICE_LDFLAGS)

$(OBJDIR)/tt_metal/device/%.o: tt_metal/device/%.cpp
	@mkdir -p $(@D)
	$(DEVICE_CXX) $(DEVICE_CXXFLAGS) $(STATIC_LIB_FLAGS) $(DEVICE_INCLUDES) -c -o $@ $<
