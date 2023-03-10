#include $(TT_METAL_HOME)/common/common.mk

ARCH_NAME ?= grayskull
DEVICE_TYPE=0

OUT ?= $(TT_METAL_HOME)/build/src/ckernels/gen/out

CKERNEL_DIR  = $(TT_METAL_HOME)/src/ckernels
# Generate ops functions for ckernels
OPS_YAML = $(TT_METAL_HOME)/src/meta/$(ARCH_NAME)/instructions/yaml/assembly.yaml
OPS_TARGET = $(OUT)/ckernel_ops.h
OPS_SCRIPT = ./genops_ckernel.py

KERNEL_LIST_YAML = ./$(ARCH_NAME)/kernel_list.yaml
HLKC_KERNEL_LIST_YAML = ./hlkc_kernel_list.yaml
KERNEL_MAKE_TARGET = $(OUT)/ckernel.mk

# Generate list of ckernels to be included by ckernel firmware when building
KERNEL_INC_TARGET = $(OUT)/ckernel_list.h $(OUT)/ckernel_enum.h $(OUT)/ckernel_string.h $(KERNEL_MAKE_TARGET)
KERNEL_INC_SCRIPT = ./genlist_ckernel.py

KERNEL_CMAKE_SCRIPT = ./gencmake.py
CKERNEL_FILES = $(wildcard $(CKERNEL_DIR)/$(ARCH_NAME)/src/*.cc)
CKERNELS_SRC = $(notdir $(basename $(CKERNEL_FILES)))

.PHONY: all
all: $(KERNEL_INC_TARGET) $(OPS_TARGET)

$(OUT):
	mkdir -p $(OUT)

$(OPS_TARGET): $(OUT) $(OPS_YAML) $(OPS_SCRIPT)
	@$(OPS_SCRIPT) $(OPS_YAML) $(OPS_TARGET)
	@$(PRINT_SUCCESS)

$(KERNEL_INC_TARGET): $(OUT) $(KERNEL_LIST_YAML) $(KERNEL_INC_SCRIPT)
	@$(KERNEL_INC_SCRIPT) $(KERNEL_LIST_YAML) $(HLKC_KERNEL_LIST_YAML) $(KERNEL_INC_TARGET) $(ARCH_NAME)
	echo " @$(KERNEL_INC_SCRIPT) $(KERNEL_LIST_YAML) $(HLKC_KERNEL_LIST_YAML) $(KERNEL_INC_TARGET) $(ARCH_NAME)"
	@$(PRINT_SUCCESS)

clean:
	rm -Rf $(OUT)
