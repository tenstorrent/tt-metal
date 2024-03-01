TEST_INCLUDES = -I$(TT_METAL_HOME)/tests/ -I$(TT_METAL_HOME)/tt_eager/.

include $(TT_METAL_HOME)/tests/tt_metal/tt_metal/module.mk
include $(TT_METAL_HOME)/tests/tt_eager/module.mk
include $(TT_METAL_HOME)/tests/ttnn/module.mk

TESTS_TO_BUILD = \
	tests/tt_metal \
	tests/tt_eager \
	tests/ttnn \

tests: tests/all

tests/all: $(TESTS_TO_BUILD)
