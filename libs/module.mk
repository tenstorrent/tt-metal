# Change for later when eager is split out
TT_LIBS_HOME ?= $(TT_METAL_HOME)
TT_METAL_BASE_INCLUDES = $(BASE_INCLUDES)
EAGER_OUTPUT_DIR = $(OUT)/dist

LIBS_INCLUDES = $(TT_METAL_BASE_INCLUDES) -Ilibs/

include libs/tensor/module.mk
include libs/dtx/module.mk
include libs/tt_dnn/module.mk
include libs/tt_lib/module.mk

TT_LIBS_TO_BUILD = libs/tensor \
                   libs/dtx \
                   libs/tt_dnn \
                   libs/tt_lib \


ifdef TT_METAL_ENV_IS_DEV
TT_LIBS_TO_BUILD += \
	libs/tt_lib/dev_install
endif

libs: $(TT_LIBS_TO_BUILD)

eager_package: python_env/dev
	python -m build --outdir $(EAGER_OUTPUT_DIR)

eager_package/clean:
	rm -rf libs/*.egg-info
	rm -rf $(EAGER_OUTPUT_DIR)
