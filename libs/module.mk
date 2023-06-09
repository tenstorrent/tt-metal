# Change for later when eager is split out
TT_LIBS_HOME ?= $(TT_METAL_HOME)
TT_METAL_BASE_INCLUDES = $(BASE_INCLUDES)

LIBS_INCLUDES = $(TT_METAL_BASE_INCLUDES) -Ilibs/

include libs/tensor/module.mk
include libs/dtx/module.mk
include libs/tt_dnn/module.mk
include libs/tt_lib/module.mk

TT_LIBS_TO_BUILD = libs/tensor \
                   libs/dtx \
                   libs/tt_dnn \
                   libs/tt_lib \

libs: $(TT_LIBS_TO_BUILD)
