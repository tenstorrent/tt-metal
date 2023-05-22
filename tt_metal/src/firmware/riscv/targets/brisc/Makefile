include $(TT_METAL_HOME)/tt_metal/common/common.mk

CKERNELS_DIR=$(TT_METAL_HOME)/src/ckernels
ARCH_NAME ?= grayskull
DEVICE_TYPE=0

SOURCES =\
	$(TT_METAL_HOME)/src/firmware/riscv/common/risc_common.cc \
	brisc.cc

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  SOURCES += $(TT_METAL_HOME)/src/firmware/riscv/wormhole/tdma_xmov.c
  SOURCES += $(TT_METAL_HOME)/src/firmware/riscv/wormhole/noc/noc.c
else
  SOURCES += $(TT_METAL_HOME)/src/firmware/riscv/$(ARCH_NAME)/tdma_xmov.c
  SOURCES += $(TT_METAL_HOME)/src/firmware/riscv/$(ARCH_NAME)/noc/noc.c
endif

FIRMWARE_NAME = brisc
INFO_NAME = brisc

# TODO: rk: get rid of one on TT_METAL_HOME decls
INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/targets/ncrisc \
	-I$(TT_METAL_HOME)/src/ckernels/$(ARCH_NAME)/common/inc \
	-I$(TT_METAL_HOME) \
	-I$(TT_METAL_HOME)/tt_metal

LINKER_SCRIPT_NAME = brisc.ld

# -fno-tree-loop-distribute-patterns is needed to prevent memset from being instantiated which gets forced in L0, which can get called from L1 functions
OPT_FLAGS = -flto -ffast-math -Os -g -fno-tree-loop-distribute-patterns

# OPT_FLAGS = -flto -ffast-math -O0 -g -fno-tree-loop-distribute-patterns

include ../../toolchain/riscv.mk

# Add specific defines for FW source code
# so we can set specific values in there correctly
DEFINES += -DCOMPILE_FOR_BRISC

ifeq ("$(ARCH_NAME)", "wormhole_b0")
DEFINES += -DRISC_B0_HW
endif

# Add profiling kernel define
ifneq ("$(PROFILE_KERNEL)", "")
DEFINES += -D PROFILE_KERNEL=$(PROFILE_KERNEL)
endif

ifneq ("$(NOC_INDEX)", "")
DEFINES += -D NOC_INDEX=$(NOC_INDEX)
endif

KERNEL_COMPILE_TIME_ARGS := $(filter KERNEL_COMPILE_TIME_ARG_%,$(.VARIABLES))
DEFINES += $(foreach var,$(KERNEL_COMPILE_TIME_ARGS),-D$(var)=$($(var)))

clean:
	rm -rfd $(OUTPUT_DIR)

linker_scripts: $(LINKER_SCRIPT)

all:: $(OUTPUT_DIR)/brisc.hex $(OUTPUT_DIR)/brisc.map $(OUTPUT_DIR)/brisc.fwlog $(OUTPUT_DIR)/brisc-decodedline.txt $(OUTPUT_DIR)/brisc-debuginfo.txt $(OUTPUT_DIR)/brisc-symbols.txt
