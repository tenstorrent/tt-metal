TEST_INCLUDES = -I$(TT_METAL_HOME)/test/

# only include these modules if we're in development
ifdef TT_METAL_ENV_IS_DEV
include $(TT_METAL_HOME)/test/tt_metal/build_kernels_for_riscv/module.mk
include $(TT_METAL_HOME)/test/tt_metal/llrt/module.mk
include $(TT_METAL_HOME)/test/tt_metal/tt_metal/module.mk
endif

TESTS_TO_BUILD = \
	build_kernels_for_riscv/tests \
	llrt/tests \
	tt_metal/tests

test: test/all

test/all: $(TESTS_TO_BUILD)
