#include $(BUDA_HOME)/common/common.mk

ARCH_NAME ?= grayskull
DEVICE_TYPE=0

OUT ?= $(BUDA_HOME)/build/src/ckernels/gen/out

CKERNEL_DIR  = $(BUDA_HOME)/src/ckernels
# Generate ops functions for ckernels
OPS_YAML = $(BUDA_HOME)/src/meta/$(ARCH_NAME)/instructions/yaml/assembly.yaml
OPS_TARGET = $(OUT)/ckernel_ops.h
OPS_SCRIPT = ./genops_ckernel.py

KERNEL_LIST_YAML = ./$(ARCH_NAME)/kernel_list.yaml
HLKC_KERNEL_LIST_YAML = ./hlkc_kernel_list.yaml
KERNEL_MAKE_TARGET = $(OUT)/ckernel.mk
KERNEL_BAZEL_HELPER = $(OUT)/ckernel.bzl
KERNEL_CMAKE_TARGET_SRC = $(CKERNEL_DIR)/$(ARCH_NAME)/src/CMakeLists.txt

GTKW_CKERNELS = $(OUT)/ckernel.gtkw_filter

# Generate list of ckernels to be included by ckernel firmware when building
KERNEL_INC_TARGET = $(OUT)/ckernel_list.h $(OUT)/ckernel_enum.h $(OUT)/ckernel_string.h $(KERNEL_MAKE_TARGET) $(GTKW_CKERNELS) $(KERNEL_BAZEL_HELPER)
KERNEL_INC_SCRIPT = ./genlist_ckernel.py


KERNEL_CMAKE_SCRIPT = ./gencmake.py
CKERNEL_FILES = $(wildcard $(CKERNEL_DIR)/$(ARCH_NAME)/src/*.cc)
CKERNELS_SRC = $(notdir $(basename $(CKERNEL_FILES)))

.PHONY: all
all: $(KERNEL_INC_TARGET) $(KERNEL_CMAKE_TARGET_SRC) $(OPS_TARGET)

$(OUT):
	mkdir -p $(OUT)

$(OPS_TARGET): $(OUT) $(OPS_YAML) $(OPS_SCRIPT)
	@$(OPS_SCRIPT) $(OPS_YAML) $(OPS_TARGET)
	@$(PRINT_SUCCESS)

$(KERNEL_INC_TARGET): $(OUT) $(KERNEL_LIST_YAML) $(KERNEL_INC_SCRIPT)
	@$(KERNEL_INC_SCRIPT) $(KERNEL_LIST_YAML) $(HLKC_KERNEL_LIST_YAML) $(KERNEL_INC_TARGET) $(ARCH_NAME)
	echo " @$(KERNEL_INC_SCRIPT) $(KERNEL_LIST_YAML) $(HLKC_KERNEL_LIST_YAML) $(KERNEL_INC_TARGET) $(ARCH_NAME)"
	@$(PRINT_SUCCESS)

$(KERNEL_CMAKE_TARGET_SRC): $(KERNEL_CMAKE_SCRIPT) $(KERNEL_LIST_YAML) $(OUT)
	@$(KERNEL_CMAKE_SCRIPT) $(KERNEL_LIST_YAML) $(KERNEL_CMAKE_TARGET_SRC)

clean:
	rm -Rf $(OUT)
