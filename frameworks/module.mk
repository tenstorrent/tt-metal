include $(TT_METAL_HOME)/frameworks/tt_dispatch/module.mk

FRAMEWORKS_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/frameworks/

TT_FRAMEWORKS_TO_BUILD = frameworks/tt_dispatch

frameworks: $(TT_FRAMEWORKS_TO_BUILD)
