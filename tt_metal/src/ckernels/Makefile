
CKERNELS_DIR = $(TT_METAL_HOME)/src/ckernels
FIRMWARE_RISCV_DIR  = $(TT_METAL_HOME)/tt_metal/src/firmware/riscv
COMPILED_KERNELS_DIR?=$(CKERNELS_DIR)/$(ARCH_NAME)/src/out
COMPILED_KERNELS_DIR_FINAL= $(COMPILED_KERNELS_DIR)/
UNUSED_COMPILED_KERNELS_DIR=$(CKERNELS_DIR)/$(ARCH_NAME)/src/blank/out/
CKERNELS_COMMON_DIR=$(CKERNELS_DIR)/$(ARCH_NAME)/common/

CKERNELS_COMMON_OUT_DIR ?= $(CKERNELS_DIR)/$(ARCH_NAME)/common/out/
CLEAN_OUTPUT_DIR ?= 1

export CKERNELS = $(CKERNELS_DIR)

# Adjust these to only include kernels that will be used in the test
KERNELS ?=
UPCASE_KERNELS = $(shell echo $(KERNELS) | tr a-z A-Z)
KERNEL_DEFINES = $(addprefix -DUCK_, $(UPCASE_KERNELS))
COMPILED_KERNELS ?= $(addsuffix .o, $(addprefix $(COMPILED_KERNELS_DIR_FINAL), $(KERNELS)))
UNUSED_KERNELS ?=
UNUSED_COMPILED_KERNELS ?= $(addsuffix .o, $(addprefix $(COMPILED_KERNELS_DIR_FINAL), $(UNUSED_KERNELS)))
EXTRA_OBJECTS = $(COMPILED_KERNELS) $(UNUSED_COMPILED_KERNELS)
PERF_DUMP ?= 0
PERF_DUMP_LEVEL ?= 0
SPILL_DRAM ?= 0

DEFINES = $(KERNEL_DEFINES)

ifeq ($(PERF_DUMP), 1)
DEFINES += $(addprefix -DPERF_DUMP=, $(PERF_DUMP))
DEFINES += $(addprefix -DPERF_DUMP_LEVEL=, $(PERF_DUMP_LEVEL))
endif

ifeq ($(SPILL_DRAM), 1)
DEFINES += $(addprefix -DSPILL_DRAM=, $(SPILL_DRAM))
endif

DISABLE_FWLOG ?= 1

# provide in command line or kernel makefile
SOURCES ?=
SOURCES += $(wildcard $(CKERNELS_COMMON_OUT_DIR)/*.o)
FIRMWARE_NAME ?=
INFO_NAME = firmware
ifeq ($(TEST),)
$(error TEST must be specified on the command line)
endif

ifeq ($(FIRMWARE_NAME),)
$(error FIRMWARE_NAME must be specified on the command line)
endif

ifeq ($(OUTPUT_DIR),)
$(error OUTPUT_DIR must be specified on the command line)
endif

LINKER_SCRIPT_NAME ?= trisc0.ld
#Format Success String
SUCCESS_STRING=\033[7;32m[SUCCESS]\033[0m
PRETTY_2_COL = awk '{ printf "%-50s %-10s\n",$$1, $$2; }'
PRINT_SUCCESS =  printf "$@ $(SUCCESS_STRING)\n" | $(PRETTY_2_COL)


.PHONY: build
build :
ifeq ($(CLEAN_OUTPUT_DIR),1)
	rm -Rf $(OUTPUT_DIR)  # if running from an HLKC build, OUTPUT_DIR is already cleared during common compile
	@mkdir -p $(OUTPUT_DIR)
endif
	@printf "%s" '$(KERNELS)' > $(OUTPUT_DIR)/kernel_list.txt
	$(MAKE) all

include $(FIRMWARE_RISCV_DIR)/toolchain/riscv_link_only.mk

ifeq ($(MAKE_FW_MAP),1)
all :: $(OUTPUT_DIR)/$(FIRMWARE_NAME).hex $(OUTPUT_DIR)/$(FIRMWARE_NAME).map $(OUTPUT_DIR)/ckernel.fwlog $(OUTPUT_DIR)/firmware-decodedline.txt $(OUTPUT_DIR)/firmware-debuginfo.txt $(OUTPUT_DIR)/firmware-symbols.txt $(OUTPUT_DIR)/$(FIRMWARE_NAME).map
else
all :: $(OUTPUT_DIR)/$(FIRMWARE_NAME).hex $(OUTPUT_DIR)/ckernel.fwlog $(OUTPUT_DIR)/firmware-decodedline.txt $(OUTPUT_DIR)/firmware-debuginfo.txt $(OUTPUT_DIR)/firmware-symbols.txt
endif

linker_scripts: $(LINKER_SCRIPT)

clean :
	rm -rfd $(OUTPUT_DIR)
