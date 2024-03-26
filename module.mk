CONFIG ?= assert
ENABLE_PROFILER ?= 0
ENABLE_TRACY ?= 0
ENABLE_CODE_TIMERS ?= 0
# TODO: enable OUT to be per config (this impacts all scripts that run tests)
# OUT ?= build_$(DEVICE_RUNNER)_$(CONFIG)
OUT ?= $(TT_METAL_HOME)/build
PREFIX ?= $(OUT)

# Disable by default, use negative instead for consistency with BBE
TT_METAL_VERSIM_DISABLED ?= 1

CONFIG_CFLAGS =
CONFIG_LDFLAGS =

# For production builds so the final pybinded so has all binaries + symbols
TT_METAL_CREATE_STATIC_LIB ?= 0

ifeq ($(CONFIG), release)
CONFIG_CFLAGS += -O3
else ifeq ($(CONFIG), ci)  # significantly smaller artifacts
CONFIG_CFLAGS += -O3 -DDEBUG=DEBUG
CONFIG_LDFLAGS += -Wl,--verbose
else ifeq ($(CONFIG), assert)
CONFIG_CFLAGS += -O3 -g -DDEBUG=DEBUG
else ifeq ($(CONFIG), asan)
CONFIG_CFLAGS += -O3 -g -DDEBUG=DEBUG -fsanitize=address
CONFIG_LDFLAGS += -fsanitize=address
else ifeq ($(CONFIG), ubsan)
CONFIG_CFLAGS += -O3 -g -DDEBUG=DEBUG -fsanitize=undefined
CONFIG_LDFLAGS += -fsanitize=undefined
else ifeq ($(CONFIG), debug)
CONFIG_CFLAGS += -O0 -g -DDEBUG=DEBUG
else
$(error Unknown value for CONFIG "$(CONFIG)")
endif

ifeq ($(TT_METAL_VERSIM_DISABLED),0)
  UMD_VERSIM_STUB = 0
else
  # Need to always define this versim disabled flag for cpp
  CONFIG_CFLAGS += -DTT_METAL_VERSIM_DISABLED
  UMD_VERSIM_STUB = 1
endif
ifeq ($(ENABLE_CODE_TIMERS), 1)
CONFIG_CFLAGS += -DTT_ENABLE_CODE_TIMERS
endif

# Gate certain dev env requirements behind this
ifeq ("$(TT_METAL_ENV)", "dev")
TT_METAL_ENV_IS_DEV = 1
endif

OBJDIR 		= $(OUT)/obj
LIBDIR 		= $(OUT)/lib
BINDIR 		= $(OUT)/bin
INCDIR 		= $(OUT)/include
TESTDIR     = $(OUT)/test
DOCSDIR     = $(OUT)/docs
TOOLS = $(OUT)/tools

UMD_HOME = $(TT_METAL_HOME)/tt_metal/third_party/umd
UMD_VERSIM_HEADERS = $(TT_METAL_VERSIM_ROOT)/versim/
UMD_USER_ROOT = $(TT_METAL_HOME)
# Top level flags, compiler, defines etc.

ifeq ("$(ARCH_NAME)", "wormhole_b0")
	BASE_INCLUDES=-Itt_metal/hw/inc -Itt_metal/hw/inc/wormhole -Itt_metal/hw/inc/wormhole/wormhole_b0_defines -I$(UMD_HOME)/src/firmware/riscv/wormhole
else
	BASE_INCLUDES=-Itt_metal/hw/inc -Itt_metal/hw/inc/$(ARCH_NAME) -I$(UMD_HOME)/src/firmware/riscv/$(ARCH_NAME)
endif

# TODO: rk reduce this to one later
BASE_INCLUDES+=-I./ -I./tt_metal/ -I$(UMD_HOME)/

#WARNINGS ?= -Wall -Wextra
WARNINGS ?= -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter
CC ?= gcc
CXX ?= g++
CFLAGS ?= -MMD $(WARNINGS) -I. $(CONFIG_CFLAGS) -mavx2 -DBUILD_DIR=\"$(OUT)\"
CXXFLAGS ?= --std=c++17 -fvisibility-inlines-hidden -Werror

LDFLAGS ?= $(CONFIG_LDFLAGS) -L$(LIBDIR) \
	-ldl \
	-lz \
	-lboost_thread \
	-lboost_filesystem \
	-lboost_system \
	-lboost_regex \
	-lpthread \
	-latomic \
	-lhwloc
ifdef TT_METAL_ENV_IS_DEV
LDFLAGS += \
	-Wl,-rpath,$(PREFIX)/lib
endif
SHARED_LIB_FLAGS = -shared -fPIC
STATIC_LIB_FLAGS = -fPIC
ifeq ($(findstring clang,$(CC)),clang)
WARNINGS += -Wno-c++11-narrowing -Wno-c++2a-extensions
LDFLAGS += -lstdc++
else
WARNINGS += -Wmaybe-uninitialized
LDFLAGS += -lstdc++
endif

# For GDDR5 bug in WH
ifneq (,$(filter "$(ARCH_NAME)","wormhole" "wormhole_b0"))
	ISSUE_3487_FIX = 1
endif

set_up_kernels:
	$(MAKE) -f $(TT_METAL_HOME)/tt_metal/hw/Makefile-runtime -C $(TT_METAL_HOME)/tt_metal/hw prepare

set_up_kernels/clean:
	$(MAKE) -f $(TT_METAL_HOME)/tt_metal/hw/Makefile-runtime -C $(TT_METAL_HOME)/tt_metal/hw clean

ifeq ($(ENABLE_PROFILER), 1)
CFLAGS += -DPROFILER
endif

ifeq ($(ENABLE_TRACY), 1)
CFLAGS += -DTRACY_ENABLE -fno-omit-frame-pointer -fPIC
LDFLAGS += -ltracy -rdynamic
endif

LIBS_TO_BUILD =
ifdef TT_METAL_ENV_IS_DEV
LIBS_TO_BUILD += \
	python_env/dev \
	git_hooks
endif

LIBS_TO_BUILD += \
	tracy \
	set_up_kernels \
	umd_device \
	tools \
	tt_metal \
	tracy \
	tt_eager \
	ttnn

# These must be in dependency order (enforces no circular deps)
include $(UMD_HOME)/device/module.mk
include $(TT_METAL_HOME)/tt_metal/common/common.mk
include $(TT_METAL_HOME)/tt_metal/module.mk
include $(TT_METAL_HOME)/tt_eager/module.mk
include $(TT_METAL_HOME)/ttnn/module.mk
include $(TT_METAL_HOME)/tt_metal/python_env/module.mk
include $(TT_METAL_HOME)/tests/module.mk

# only include these modules if we're in development
ifdef TT_METAL_ENV_IS_DEV
include $(TT_METAL_HOME)/infra/git_hooks/module.mk
endif

build: $(LIBS_TO_BUILD)

clean: set_up_kernels/clean eager_package/clean
	test -d build && find build  -mindepth 1 -maxdepth 1 ! -path "build/python_env" -exec rm -rf {} + || true
	rm -rf dist/

nuke: clean python_env/clean
	rm -rf $(OUT)
