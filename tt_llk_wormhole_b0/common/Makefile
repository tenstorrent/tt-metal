include $(BUDA_HOME)/infra/common.mk

ARCH_NAME ?= wormhole_b0
DEVICE_TYPE=0

MAILBOX_ADDR ?= 4
HLKC_KERNELS ?=
KERNELS ?= cblank
UPCASE_KERNELS = $(shell echo $(KERNELS) | tr a-z A-Z)
KERNEL_DEFINES = $(addprefix -DUCK_, $(UPCASE_KERNELS))
MAILBOX_ADDR_DEFINE = $(addprefix -DMAILBOX_ADDR=, $(MAILBOX_ADDR))
DEFINES = $(KERNEL_DEFINES) $(MAILBOX_ADDR_DEFINE)
PERF_DUMP ?= 0
PERF_DUMP_LEVEL ?= 0
INTERMED_DUMP ?= 0
PERF_DUMP_CONCURRENT ?= 0

ifeq ($(PERF_DUMP), 1)
DEFINES += $(addprefix -DPERF_DUMP=, $(PERF_DUMP))
endif
ifeq ($(INTERMED_DUMP), 1)
DEFINES += $(addprefix -DINTERMED_DUMP=, $(INTERMED_DUMP))
endif
ifeq ($(PERF_DUMP_CONCURRENT), 1)
DEFINES += $(addprefix -DPERF_DUMP_CONCURRENT=, $(PERF_DUMP_CONCURRENT))
endif
DEFINES += $(addprefix -DPERF_DUMP_LEVEL=, $(PERF_DUMP_LEVEL)) -DARCH_WORMHOLE

CKERNEL_DIR  = $(BUDA_HOME)/src/ckernels
CKERNEL_INC_DIR  = $(BUDA_HOME)/build/src/ckernels
FIRMWARE_RISCV_DIR = $(BUDA_HOME)/src/firmware/riscv
FIRMWARE_RISCV_INC_DIR = $(BUDA_HOME)/build/src/firmware/riscv

ifdef HLKC_INC
# if HLKC_INC is defined, we have detected the HLKC flow
# these are the includes for the HLKC auto-generated flow
INCLUDES = \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/common/inc \
	-I$(HLKC_INC) \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/llk_lib/ \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/ \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/noc \
	-I$(BUDA_HOME)/perf_lib/ \
	-I$(CKERNEL_INC_DIR)/gen/out # needed for ckernel_ops.h
else
# these are the includes for the TB LLK flow
INCLUDES = \
	-I$(CKERNEL_DIR)/test/inc \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/common/inc \
	-I$(CKERNEL_DIR)/hlkc/inc \
	-I$(CKERNEL_DIR)/hlkc/src \
	-I$(CKERNEL_DIR)/$(ARCH_NAME)/llk_lib/ \
	-I$(CKERNEL_INC_DIR)/gen/out \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME) \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/noc \
	-I$(BUDA_HOME)/perf_lib/
endif

DISABLE_FWLOG ?= 1

# provide in command line or kernel makefile
RISCV_TOOLCHAIN_DIR = $(BUDA_HOME)/src/firmware/riscv/toolchain
SOURCES ?= 
SOURCES += $(wildcard src/*.cc)
OUTPUT_DIR ?= out

.PHONY: build
build : $(OUTPUT_DIR)
	rm -Rf $(OUTPUT_DIR)  # always clean
	@mkdir -p $(OUTPUT_DIR)
	$(MAKE) ck_common_prereqs
	@$(PRINT_SUCCESS)

include $(FIRMWARE_RISCV_DIR)/toolchain/riscv.mk
ck_common_prereqs: $(OUTPUT_DIR) $(OUTPUT_DIR)/ckernel.o $(OUTPUT_DIR)/ckernel_template.o $(OUTPUT_DIR)/ckernel_main.o $(OUTPUT_DIR)/ckernel_perf_unpack_pack.o $(OUTPUT_DIR)/ckernel.fwlog $(EXTRA_OBJECTS)

clean:
	rm -rf out
