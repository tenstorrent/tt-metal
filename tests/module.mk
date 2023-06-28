TEST_INCLUDES = -I$(TT_METAL_HOME)/tests/ -I$(TT_METAL_HOME)/libs/.

include $(TT_METAL_HOME)/tests/tt_metal/build_kernels_for_riscv/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/llrt/module.mk
include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/module.mk

TESTS_TO_BUILD = \
	tests/build_kernels_for_riscv \
	tests/llrt \
	tests/tt_metal

tests: tests/all

tests/all: $(TESTS_TO_BUILD)
