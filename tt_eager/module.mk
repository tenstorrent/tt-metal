# Change for later when eager is split out
TT_LIBS_HOME ?= $(TT_METAL_HOME)
TT_METAL_BASE_INCLUDES = $(BASE_INCLUDES)
EAGER_OUTPUT_DIR = $(OUT)/dist

TT_EAGER_INCLUDES = $(TT_METAL_BASE_INCLUDES) -Itt_eager/

include tt_eager/tensor/module.mk
include tt_eager/dtx/module.mk
include tt_eager/tt_dnn/module.mk
include tt_eager/queue/module.mk
include tt_eager/tt_lib/module.mk

TT_LIBS_TO_BUILD = tt_eager/tensor \
                   tt_eager/dtx \
                   tt_eager/tt_dnn \
                   tt_eager/queue \
                   tt_eager/tt_lib \


ifdef TT_METAL_ENV_IS_DEV
TT_LIBS_TO_BUILD += \
	$(TT_LIB_LIB_LOCAL_SO)
endif

tt_eager: $(TT_LIBS_TO_BUILD)

eager_package: python_env/dev
	source build/python_env/bin/activate
	python -m build --outdir $(EAGER_OUTPUT_DIR)

eager_package/clean:
	rm -rf tt_eager/*.egg-info
	rm -rf $(EAGER_OUTPUT_DIR)
