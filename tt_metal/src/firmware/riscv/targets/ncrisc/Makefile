include $(TT_METAL_HOME)/tt_metal/common/common.mk

CKERNELS_DIR=$(TT_METAL_HOME)/src/ckernels
ARCH_NAME ?= grayskull
DEVICE_TYPE=0

PERF_DUMP ?= 0
INTERMED_DUMP ?= 0
PERF_DUMP_LEVEL ?= 0

DEFINES =

SOURCES = \
	ncrisc.cc \
	context.cc \
	$(TT_METAL_HOME)/src/firmware/riscv/common/risc_common.cc \


ifeq ($(PERF_DUMP), 1)
SOURCES += risc_perf.cc
endif

ifeq ("$(ARCH_NAME)","wormhole_b0")
SOURCES += $(TT_METAL_HOME)/src/firmware/riscv/wormhole/risc_chip_specific.c
else
SOURCES += $(TT_METAL_HOME)/src/firmware/riscv/$(ARCH_NAME)/risc_chip_specific.c
endif

FIRMWARE_NAME = ncrisc
INFO_NAME = ncrisc

# TODO: rk: get rid of one on TT_METAL_HOME decls
INCLUDES = -I$(TT_METAL_HOME)/src/firmware/riscv/targets/ncrisc \
    -I$(TT_METAL_HOME)/src/firmware/riscv/$(ARCH_NAME) \
	-I$(TT_METAL_HOME)/src/firmware/riscv/common \
	-I$(TT_METAL_HOME)/src/ckernels/$(ARCH_NAME)/common/inc \
	-I$(TT_METAL_HOME) \
	-I$(TT_METAL_HOME)/tt_metal \
	-I$(KERNEL_INC)

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/wormhole/noc
else
  INCLUDES += -I$(TT_METAL_HOME)/src/firmware/riscv/$(ARCH_NAME)/noc
endif

LINKER_SCRIPT_NAME = ncrisc.ld
# TODO(agrebenisan): Figure out why ncrisc-debug.ld doesn't work
# ifeq ($(DEBUG_MODE),1)
# 	LINKER_SCRIPT_NAME = ncrisc-debug.ld
# endif

# uncomment to enable detailed logging in NCRISC post code for waveform debug
# DEFINES = -D ENABLE_NCRISC_DEBUG_POST_CODES

ifeq ($(PERF_DUMP), 1)
DEFINES += $(addprefix -DPERF_DUMP=, $(PERF_DUMP))
DEFINES += $(addprefix -DPERF_DUMP_LEVEL=, $(PERF_DUMP_LEVEL))
endif

ifeq ($(INTERMED_DUMP), 1)
DEFINES += $(addprefix -DINTERMED_DUMP=, $(INTERMED_DUMP))
endif

ifeq ("$(ARCH_NAME)", "wormhole_b0")
DEFINES += -DRISC_B0_HW
endif

# -fno-tree-loop-distribute-patterns is needed to prevent memset from being instantiated which gets forced in L0, which can get called from L1 functions
#  rkim: above flag causes hangs when deasserting reset on ncrisc, removing for now
OPT_FLAGS = -flto -ffast-math -Os -g # -fno-tree-loop-distribute-patterns

include ../../toolchain/riscv.mk

# Add specific defines for FW source code
# so we can set specific values in there correctly
DEFINES += -DCOMPILE_FOR_NCRISC

# Add profiling kernel define
ifneq ("$(PROFILE_KERNEL)", "")
DEFINES += -D PROFILE_KERNEL=$(PROFILE_KERNEL)
endif

ifneq ("$(NOC_INDEX)", "")
DEFINES += -D NOC_INDEX=$(NOC_INDEX)
endif

KERNEL_COMPILE_TIME_ARGS := $(filter KERNEL_COMPILE_TIME_ARG_%,$(.VARIABLES))
DEFINES += $(foreach var,$(KERNEL_COMPILE_TIME_ARGS),-D$(var)=$($(var)))

linker_scripts: $(LINKER_SCRIPT)

all:: $(OUTPUT_DIR)/ncrisc.hex $(OUTPUT_DIR)/ncrisc.map $(OUTPUT_DIR)/ncrisc.fwlog $(OUTPUT_DIR)/ncrisc-decodedline.txt $(OUTPUT_DIR)/ncrisc-debuginfo.txt $(OUTPUT_DIR)/ncrisc-symbols.txt

clean::
	rm -rfd $(OUTPUT_DIR)/*
