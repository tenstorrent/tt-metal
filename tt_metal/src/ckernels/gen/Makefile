#include $(TT_METAL_HOME)/tt_metal/common/common.mk

ARCH_NAME ?= grayskull

OUT ?= $(TT_METAL_HOME)/build/src/ckernels/gen/out

# Generate ops functions for ckernels
OPS_YAML = $(TT_METAL_HOME)/tt_metal/src/meta/$(ARCH_NAME)/instructions/yaml/assembly.yaml
OPS_TARGET = $(OUT)/ckernel_ops.h
OPS_SCRIPT = ./genops_ckernel.py

.PHONY: all
all: $(OPS_TARGET)

$(OUT):
	mkdir -p $(OUT)

$(OPS_TARGET): $(OUT) $(OPS_YAML) $(OPS_SCRIPT)
	@$(OPS_SCRIPT) $(OPS_YAML) $(OPS_TARGET)
	@$(PRINT_SUCCESS)

clean:
	rm -Rf $(OUT)
