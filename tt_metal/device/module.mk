# Every variable in subdir must be prefixed with subdir (emulating a namespace)
DEVICE_LIB = $(LIBDIR)/libdevice.a
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

# If versim enabled, the following needs to be provided
TT_METAL_VERSIM_FULL_DUMP ?= 0
ifeq ($(TT_METAL_VERSIM_DISABLED),1)
  DEVICE_SRCS += tt_metal/device/tt_versim_stub.cpp
else
  DEVICE_SRCS += tt_metal/device/tt_versim_device.cpp
  ifndef TT_METAL_VERSIM_ROOT
    $(error VERSIM enabled but TT_METAL_VERSIM_ROOT not defined)
  endif
endif

DEVICE_OBJS = $(addprefix $(OBJDIR)/, $(DEVICE_SRCS:.cpp=.o))
DEVICE_DEPS = $(addprefix $(OBJDIR)/, $(DEVICE_SRCS:.cpp=.d))

DEVICE_INCLUDES=      	\
  -DFMT_HEADER_ONLY     \
  $(COMMON_INCLUDES)    \
  -I$(TT_METAL_HOME)/tt_metal/third_party/fmt

ifeq ($(TT_METAL_VERSIM_FULL_DUMP), 1)
ifneq ("$(ARCH_NAME)", "wormhole_b0")
  $(error "FPU wave dump only available for wormhole_b0")
endif
TT_METAL_VERSIM_LIB_DIR = $(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/fpu_waves_lib
else
TT_METAL_VERSIM_LIB_DIR = $(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/lib
endif

ifeq ($(TT_METAL_VERSIM_DISABLED),0)
DEVICE_INCLUDES+=      	\
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/vendor/tenstorrent-repositories/verilator/include         \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/vendor/tenstorrent-repositories/verilator/include/vltstd  \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/vendor/yaml-cpp/include                                   \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/vendor/fc4sc/includes                                     \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/vendor/tclap/include                                      \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/vendor/tenstorrent-repositories/range-v3/include          \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/hardware/tdma/tb/tile                                 \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/firmware/riscv/common                                 \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/meta/$(ARCH_NAME)/instructions/inc                       \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/meta/$(ARCH_NAME)/types/inc                              \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/software/command_assembler/inc                        \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/t6ifc/common                                          \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/t6ifc/versim-core                                     \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/t6ifc/versim-core/common                              \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/t6ifc/versim-core/common/inc                          \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/t6ifc/versim-core/monitors                            \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/t6ifc/versim-core/checkers                            \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/src/tvm/inc                                               \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers/usr_include                                               \
  -I$(TT_METAL_VERSIM_ROOT)/versim/$(ARCH_NAME)/headers

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
ifeq ($(TT_METAL_VERSIM_DISABLED),0)
DEVICE_LDFLAGS = \
  -Wl,-rpath,$(TT_METAL_HOME)/tt_metal/third_party/common_lib,-rpath,$(TT_METAL_VERSIM_LIB_DIR),-rpath,$(TT_METAL_VERSIM_ROOT)/required_libraries \
  -L$(TT_METAL_VERSIM_LIB_DIR) \
  -L$(TT_METAL_VERSIM_ROOT)/required_libraries
endif
DEVICE_LDFLAGS += \
	-L$(TT_METAL_HOME)/tt_metal/third_party/common_lib \
	-lz \
	-lpthread \
	-latomic \
	-lcommon

ifeq ($(TT_METAL_VERSIM_DISABLED),0)
  DEVICE_LDFLAGS += \
    -l:libboost_system.so.1.65.1 \
    -l:libboost_filesystem.so.1.65.1 \
    -l:libicudata.so.60 \
    -l:libicui18n.so.60 \
    -l:libicuuc.so.60 \
    -l:libboost_thread.so.1.65.1 \
    -l:libboost_regex.so.1.65.1 \
    -l:versim-core.so \
    -l:libc_verilated_hw.so
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
DEVICE_CXXFLAGS += -O3 -g -DDEBUG
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

ifeq ($(TT_METAL_VERSIM_DISABLED),1)
  DEVICE_CXXFLAGS += -DTT_METAL_VERSIM_DISABLED
endif
ifeq ($(ISSUE_3487_FIX), 1)
  DEVICE_CXXFLAGS += -DISSUE_3487_FIX
endif


-include $(DEVICE_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
device: $(DEVICE_LIB)


$(DEVICE_LIB): $(COMMON_LIB) $(DEVICE_OBJS)
	@mkdir -p $(LIBDIR)
ifeq ($(TT_METAL_VERSIM_DISABLED),0)
  # Need to build everything and link with versim if it is included
	$(DEVICE_CXX) $(DEVICE_CXXFLAGS) $(SHARED_LIB_FLAGS) -o $(DEVICE_LIB) $^ $(LDFLAGS) $(DEVICE_LDFLAGS)
else
  # Device can be static now when running without versim since we use compile-time flags for TT_METAL_ARCH
  # anyway
	ar rcs -o $@ $(DEVICE_OBJS)
endif

$(OBJDIR)/tt_metal/device/%.o: tt_metal/device/%.cpp
	@mkdir -p $(@D)
	$(DEVICE_CXX) $(DEVICE_CXXFLAGS) $(STATIC_LIB_FLAGS) $(DEVICE_INCLUDES) -c -o $@ $<
