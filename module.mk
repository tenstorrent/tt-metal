CONFIG ?= assert
BACKEND_PROFILER_EN ?= 0
ENABLE_CODE_TIMERS ?= 0
# TODO: enable OUT to be per config (this impacts all scripts that run tests)
# OUT ?= build_$(DEVICE_RUNNER)_$(CONFIG)
OUT ?= $(TT_METAL_HOME)/build
PREFIX ?= $(OUT)

# Disable by default, use negative instead for consistency with BBE
DISABLE_VERSIM_BUILD ?= 1

CONFIG_CFLAGS =
CONFIG_LDFLAGS =

ifeq ($(CONFIG), release)
CONFIG_CFLAGS += -O3 -fno-lto
else ifeq ($(CONFIG), ci)  # significantly smaller artifacts
CONFIG_CFLAGS += -O3 -DDEBUG=DEBUG
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

ifeq ($(BACKEND_PROFILER_EN), 1)
CONFIG_CFLAGS += -DBACKEND_PERF_PROFILER
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

# Top level flags, compiler, defines etc.

ifeq ("$(ARCH_NAME)", "wormhole_b0")
	BASE_INCLUDES=-Itt_metal/src/firmware/riscv/wormhole -Itt_metal/src/firmware/riscv/wormhole/wormhole_b0_defines
else ifeq ("$(ARCH_NAME)", "wormhole")
	BASE_INCLUDES=-Itt_metal/src/firmware/riscv/wormhole -Itt_metal/src/firmware/riscv/wormhole/wormhole_a0_defines
else
	BASE_INCLUDES=-Itt_metal/src/firmware/riscv/$(ARCH_NAME)
endif

# TODO: rk reduce this to one later
BASE_INCLUDES+=-I./ -I./tt_metal/

#WARNINGS ?= -Wall -Wextra
WARNINGS ?= -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter
CC ?= gcc
CXX ?= g++
CFLAGS ?= -MMD $(WARNINGS) -I. $(CONFIG_CFLAGS) -mavx2 -DBUILD_DIR=\"$(OUT)\"
CXXFLAGS ?= --std=c++17 -fvisibility-inlines-hidden -Werror
LDFLAGS ?= $(CONFIG_LDFLAGS) -Wl,-rpath,$(PREFIX)/lib -L$(LIBDIR)/tools -L$(LIBDIR) -ldl
SHARED_LIB_FLAGS = -shared -fPIC
STATIC_LIB_FLAGS = -fPIC
ifeq ($(findstring clang,$(CC)),clang)
WARNINGS += -Wno-c++11-narrowing
LDFLAGS += -lstdc++
else
WARNINGS += -Wmaybe-uninitialized
LDFLAGS += -lstdc++
endif

LIBS_TO_BUILD = \
	common \
	src/ckernels \
	src/firmware \
	build_kernels_for_riscv \
	device \
	llrt \
	tt_metal \
	python_env \
	libs \
	tools

ifdef TT_METAL_ENV_IS_DEV
LIBS_TO_BUILD += \
	python_env/dev \
	git_hooks
endif

# These must be in dependency order (enforces no circular deps)
include $(TT_METAL_HOME)/tt_metal/common/common.mk
include $(TT_METAL_HOME)/tt_metal/module.mk
include $(TT_METAL_HOME)/libs/module.mk

# only include these modules if we're in development
ifdef TT_METAL_ENV_IS_DEV
include $(TT_METAL_HOME)/infra/git_hooks/module.mk
include $(TT_METAL_HOME)/tests/module.mk
endif

build: $(LIBS_TO_BUILD)

clean: src/ckernels/clean
	rm -rf $(OUT)
