
KERNEL_PARAM_OVERRIDE_DIR ?=
TEST_NAME ?=
KERNELS ?= cblank
UPCASE_KERNELS = $(shell echo $(KERNELS) | tr a-z A-Z)
KERNEL_DEFINES = $(addprefix -DUCK_, $(UPCASE_KERNELS))
DEFINES = $(KERNEL_DEFINES)
PERF_DUMP ?= 0


ifeq ($(PERF_DUMP), 1)
DEFINES += $(addprefix -DPERF_DUMP=, $(PERF_DUMP))
endif

ARCH_NAME ?= grayskull
DEVICE_TYPE=0

CKERNEL_DIR  = $(TT_METAL_HOME)/src/ckernels/
CKERNEL_INC_DIR  = $(TT_METAL_HOME)/build/src/ckernels/
CKERNEL_INC_OVERRIDE_DIR  ?= $(TT_METAL_HOME)/build/src/ckernels/
OUTPUT_DIR = out
FIRMWARE_RISCV_DIR = $(TT_METAL_HOME)/src/firmware/riscv
FIRMWARE_RISCV_INC_DIR = $(TT_METAL_HOME)/build/src/firmware/riscv
INCLUDES = \
	-I$(CKERNEL_INC_OVERRIDE_DIR)/$(ARCH_NAME)/common/inc \
	-I$(KERNEL_PARAM_OVERRIDE_DIR)/ \
	-I$(TT_METAL_HOME)/build/tests/$(TEST_NAME) \
	-I$(CKERNEL_INC_OVERRIDE_DIR)/$(ARCH_NAME)/src/test_params/ \
	-I$(CKERNEL_INC_DIR)/gen/out \
	-I$(CKERNEL_INC_OVERRIDE_DIR)/$(ARCH_NAME)/llk_lib/ \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME) \
	-I$(FIRMWARE_RISCV_INC_DIR)/$(ARCH_NAME)/noc \

VPATH=out

.PHONY: build
build:
	rm -Rf $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR)
	$(MAKE) -C $(CKERNEL_DIR)/$(ARCH_NAME)/common KERNELS=$(KERNELS)
	$(MAKE) $(OUTPUT_DIR)/$(KERNELS).o

clean::
	rm -Rf $(OUTPUT_DIR)

-include $(OUTPUT_DIR)/$(KERNELS).d
include $(CKERNEL_INC_DIR)/gen/out/ckernel.mk
include $(FIRMWARE_RISCV_DIR)/toolchain/riscv.mk
