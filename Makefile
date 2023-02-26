.SUFFIXES:

MAKEFLAGS := --jobs=8
# nproc can result in OOM errors for specific machines and should be reworked.
#MAKEFLAGS := --jobs=$(shell nproc)

# Setup CONFIG, DEVICE_RUNNER, and out/build dirs first
BUDA_HOME ?= $(shell git rev-parse --show-toplevel)
CONFIG ?= assert
BACKEND_PROFILER_EN ?= 0
ENABLE_CODE_TIMERS ?= 0
ARCH_NAME ?= grayskull
# TODO: enable OUT to be per config (this impacts all scripts that run tests)
# OUT ?= build_$(DEVICE_RUNNER)_$(CONFIG)
OUT ?= $(BUDA_HOME)/build
PREFIX ?= $(OUT)

# Disable by default, use negative instead for consistency with BBE
DISABLE_VERSIM_BUILD ?= 1

GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD)
GIT_HASH = $(shell git describe --always --dirty)

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
ifeq ("$(GPAI_ENV)", "dev")
GPAI_ENV_IS_DEV = 1
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
	BASE_INCLUDES=-Isrc/firmware/riscv/wormhole -Isrc/firmware/riscv/wormhole/wormhole_b0_defines
else ifeq ("$(ARCH_NAME)", "wormhole")
	BASE_INCLUDES=-Isrc/firmware/riscv/wormhole -Isrc/firmware/riscv/wormhole/wormhole_a0_defines
else
	BASE_INCLUDES=-Isrc/firmware/riscv/$(ARCH_NAME)
endif

CCACHE := $(shell command -v ccache 2> /dev/null)

# Ccache setup
ifneq ($(CCACHE),)
	export CCACHE_DIR=$(HOME)/.ccache
	export CCACHE_MAXSIZE=10
	export CCACHE_UMASK=002
	export CCACHE_COMPRESS=true
	export CCACHE_NOHARDLINK=true
	export CCACHE_DIR_EXISTS=$(shell [ -d $(CCACHE_DIR) ] && echo "1")
	ifneq ($(CCACHE_DIR_EXISTS), 1)
	CCACHE_CMD=
	else
	CCACHE_CMD=$(CCACHE)
	endif
endif

BASE_INCLUDES+=-Ithird_party/yaml-cpp/include
BASE_INCLUDES+=-I./ # to be able to include modules relative to root level from any module in the project

#WARNINGS ?= -Wall -Wextra
WARNINGS ?= -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter
CC ?= $(CCACHE_CMD) gcc
CXX ?= $(CCACHE_CMD) g++
CFLAGS ?= -MMD $(WARNINGS) -I. $(CONFIG_CFLAGS) -mavx2 -DBUILD_DIR=\"$(OUT)\" -DFMT_HEADER_ONLY -Ithird_party/fmt
CXXFLAGS ?= --std=c++17 -fvisibility-inlines-hidden
LDFLAGS ?= $(CONFIG_LDFLAGS) -Wl,-rpath,$(PREFIX)/lib -L$(LIBDIR)/tools -L$(LIBDIR) -Ldevice/lib -ldl #-l/usr/include/python3.8
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
	tools/profiler \
	src/ckernels \
	hlkc hlkc/api \
	build_kernels_for_riscv \
	device \
	llrt \
	ll_buda \
	tt_gdb \
	tensor \
	compiler \
	tools \
	python_env \
	gpai

ifdef GPAI_ENV_IS_DEV
LIBS_TO_BUILD += \
	python_env/dev \
	git_hooks
endif

build: $(LIBS_TO_BUILD)

valgrind: $(VG_EXE)
	valgrind --show-leak-kinds=all \
	--leak-check=full \
	--show-leak-kinds=all \
	--track-origins=yes \
	--verbose \
	--log-file=valgrind-out.txt $(VG_ARGS) $(VG_EXE)


clean_fw: src/firmware/clean
clean: src/ckernels/clean clean_fw
	rm -rf $(OUT)

install: build
ifeq ($(PREFIX), $(OUT))
	@echo "To install you must set PREFIX, e.g."
	@echo ""
	@echo "  PREFIX=/usr CONFIG=release make install"
	@echo ""
	@exit 1
endif
	cp -r $(LIBDIR)/* $(PREFIX)/lib/
	cp -r $(INCDIR)/* $(PREFIX)/include/
	cp -r $(BINDIR)/* $(PREFIX)/bin/

package:
	tar czf spatial2_$(shell git rev-parse HEAD).tar.gz --exclude='*.d' --exclude='*.o' --exclude='*.a' --transform 's,^,spatial2_$(shell git rev-parse HEAD)/,' -C $(OUT) .

gitinfo:
	mkdir -p $(OUT)
	rm -f $(OUT)/.gitinfo
	@echo $(GIT_BRANCH) >> $(OUT)/.gitinfo
	@echo $(GIT_HASH) >> $(OUT)/.gitinfo

# These must be in dependency order (enforces no circular deps)
include common/module.mk
include tools/profiler/module.mk
include device/module.mk
include src/ckernels/module.mk
include src/firmware/module.mk
include hlkc/module.mk
include llrt/module.mk
include build_kernels_for_riscv/module.mk
include ll_buda/module.mk
include tt_gdb/module.mk # needs to compiled after llrt and ll_buda
include common/common.mk
include tensor/module.mk
include compiler/module.mk
include tools/module.mk
include python_env/module.mk
include gpai/module.mk

# only include these modules if we're in development
ifdef GPAI_ENV_IS_DEV
include build_kernels_for_riscv/tests/module.mk
include llrt/tests/module.mk
include ll_buda/tests/module.mk
include compiler/tests/module.mk
include git_hooks/module.mk
endif

# Programming examples for external users
include programming_examples/module.mk
