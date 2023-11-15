include $(BUDA_HOME)/infra/common.mk

ARCH_NAME ?= wormhole_b0
DEVICE_TYPE=0

MAILBOX_ADDR ?= 4
KERNELS ?= cblank
UPCASE_KERNELS = $(shell echo $(KERNELS) | tr a-z A-Z)
KERNEL_DEFINES = $(addprefix -DUCK_, $(UPCASE_KERNELS))
MAILBOX_ADDR_DEFINE = $(addprefix -DMAILBOX_ADDR=, $(MAILBOX_ADDR))
DEFINES = $(KERNEL_DEFINES) $(MAILBOX_ADDR_DEFINE)
PERF_DUMP ?= 0
PERF_DUMP_LEVEL ?= 0
INTERMED_DUMP ?= 0
PERF_DUMP_CONCURRENT ?= 0
OVERLAY_DECOUPLE ?= 0

ifeq ($(PERF_DUMP), 1)
DEFINES += $(addprefix -DPERF_DUMP=, $(PERF_DUMP))
endif
ifeq ($(INTERMED_DUMP), 1)
DEFINES += $(addprefix -DINTERMED_DUMP=, $(INTERMED_DUMP))
endif
ifeq ($(PERF_DUMP_CONCURRENT), 1)
DEFINES += $(addprefix -DPERF_DUMP_CONCURRENT=, $(PERF_DUMP_CONCURRENT))
endif
ifeq ($(OVERLAY_DECOUPLE), 1)
DEFINES += $(addprefix -DOVERLAY_DECOUPLE=, $(OVERLAY_DECOUPLE))
endif
DEFINES += $(addprefix -DPERF_DUMP_LEVEL=, $(PERF_DUMP_LEVEL)) -DARCH_WORMHOLE

CKERNEL_DIR  = $(BUDA_HOME)/src/ckernels
CKERNEL_INC_DIR  = $(BUDA_HOME)/build/src/ckernels
FIRMWARE_RISCV_DIR = $(BUDA_HOME)/src/firmware/riscv
FIRMWARE_RISCV_INC_DIR = $(BUDA_HOME)/build/src/firmware/riscv

ifdef HLK_INC
# if HLK_INC is defined, we have detected the HLK compilation flow
# these are the includes for the HLK auto-generated flow
# todo.pp line 57 (-I$(CKERNEL_INC_DIR)/gen/out \) might not be needed
INCLUDES = \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/common/inc \
	-I$(HLK_INC) \
	-I$(HLK_LIB_INC) \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/llk_lib/ \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/buda/llk_io/ \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/buda/llk_api/ \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/ \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/noc \
	-I$(BUDA_HOME)/perf_lib/ \
	-I$(CKERNEL_INC_DIR)/gen/out \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/hlk_defs/
else
# these are the includes for the TB LLK flow
INCLUDES = \
	-I$(CKERNEL_DIR)/test/inc \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/common/inc \
	-I$(CKERNEL_DIR)/hlkc/inc \
	-I$(CKERNEL_DIR)/hlkc/src \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/llk_lib/ \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/buda/llk_io/ \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/buda/llk_api/ \
	-I$(CKERNEL_INC_DIR)/gen/out \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME) \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/noc \
	-I$(BUDA_HOME)/perf_lib/
endif

DISABLE_FWLOG ?= 1

# provide in command line or kernel makefile
RISCV_TOOLCHAIN_DIR = $(BUDA_HOME)/src/firmware/riscv/toolchain

SOURCES ?= 

SOURCES += src/ckernel_unity.cc 

# Sw stack io
SOURCES += $(CKERNEL_DIR)/$(ARCH_NAME)/buda/llk_io/llk_io.cc

OUTPUT_DIR ?= out

.PHONY: build
build : $(OUTPUT_DIR)
	rm -Rf $(OUTPUT_DIR)  # always clean
	@mkdir -p $(OUTPUT_DIR)
	$(MAKE) ck_common_prereqs
	@$(PRINT_SUCCESS)

include $(FIRMWARE_RISCV_DIR)/toolchain/riscv.mk

# instead of building each file separately, we build them all together in ckernel_unity.cc
# if you need to add new cc file, add it to ckernel_unity.cc
ck_common_prereqs: $(OUTPUT_DIR) $(OUTPUT_DIR)/ckernel_unity.o  $(OUTPUT_DIR)/llk_io.o $(EXTRA_OBJECTS)

clean:
	rm -rf out
