# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# =========================
# Toolchain and Directories
# =========================
CXX_VERSION     := c++17
TOOL_PATH       ?= sfpi/compiler/bin

# =========================
# Architecture Selection
# =========================
ARCH            := $(if $(archname),$(archname),$(CHIP_ARCH))
ifeq ($(ARCH),wormhole)
	ARCH_NON_COMPUTE:= -mcpu=tt-wh
	ARCH_COMPUTE    := -mcpu=tt-wh-tensix
	ARCH_DEFINE     := -DARCH_WORMHOLE
	ARCH_LLK_ROOT   := tt_llk_wormhole_b0
else ifeq ($(ARCH),blackhole)
	ARCH_NON_COMPUTE:= -mcpu=tt-bh
	ARCH_COMPUTE    := -mcpu=tt-bh-tensix
	ARCH_DEFINE     := -DARCH_BLACKHOLE
	ARCH_LLK_ROOT   := tt_llk_blackhole
else ifeq ($(ARCH),quasar)
# until there is official support for quasar in SFPI fallback to BH
	ARCH_NON_COMPUTE:= -mcpu=tt-bh
	ARCH_COMPUTE    := -mcpu=tt-bh-tensix
	ARCH_DEFINE     := -DARCH_QUASAR
	ARCH_LLK_ROOT   := tt_llk_quasar
else
$(error must provide archname or CHIP_ARCH environment variable (wormhole / blackhole))
endif

BUILD_DIR       ?= /tmp/tt-llk-build

SHARED_DIR      := $(BUILD_DIR)/shared
COVERAGE_REPORT_DIR	:= $(BUILD_DIR)/coverage_report
SHARED_OBJ_DIR  := $(SHARED_DIR)/obj
SHARED_DEPS_DIR := $(SHARED_DIR)/deps
SHARED_DIS_DIR  := $(SHARED_DIR)/dis
SHARED_ELF_DIR  := $(SHARED_DIR)/elf
INFO_FILES_DIR 	:= $(BUILD_DIR)/coverage_info

TEST_DIR        := $(BUILD_DIR)/$(testname)/$(variant)
OBJ_DIR         := $(TEST_DIR)/obj
DEPS_DIR        := $(TEST_DIR)/deps
DIS_DIR         := $(TEST_DIR)/dis
ELF_DIR         := $(TEST_DIR)/elf

PROFILER_DIR    := $(BUILD_DIR)/profiler_meta/$(testname)/$(variant)

HELPERS         := helpers
RISCV_SOURCES   := $(HELPERS)/src
LINKER_SCRIPTS  := $(HELPERS)/ld
HEADER_DIR      := hw_specific/$(ARCH)/inc
RMDIR           := rm -rf

GXX             := $(TOOL_PATH)/riscv-tt-elf-g++
OBJDUMP         := $(TOOL_PATH)/riscv-tt-elf-objdump
OBJCOPY         := $(TOOL_PATH)/riscv-tt-elf-objcopy
GCOV			:= $(realpath $(TOOL_PATH)/riscv-tt-elf-gcov)
GCOV_TOOL		:= $(realpath $(TOOL_PATH)/riscv-tt-elf-gcov-tool)

# =========================
# Compiler and Linker Flags
# =========================
OPTIONS_ALL	:= -g -O3 -std=$(CXX_VERSION) -ffast-math
OPTIONS_COMPILE := -fno-use-cxa-atexit -Wall -fno-exceptions -fno-rtti -Wunused-parameter \
				   -Wfloat-equal -Wpointer-arith -Wnull-dereference -Wredundant-decls -Wuninitialized -nostdlib -fno-builtin \
				   -Wmaybe-uninitialized -DTENSIX_FIRMWARE -DENV_LLK_INFRA -DENABLE_LLK_ASSERT $(ARCH_DEFINE)
COVERAGE_OPTIONS_COMPILE := $(OPTIONS_COMPILE)

BOOT_MODE := $(if $(bootmode),$(bootmode),brisc)
ifeq ($(BOOT_MODE),brisc)
	OPTIONS_COMPILE += -DLLK_BOOT_MODE_BRISC
else ifeq ($(BOOT_MODE),trisc)
	OPTIONS_COMPILE += -DLLK_BOOT_MODE_TRISC
endif

PROFILER_BUILD := $(if $(profiler_build),$(profiler_build),false)
ifeq ($(PROFILER_BUILD),true)
	OPTIONS_COMPILE += -DLLK_PROFILER
endif

COVERAGE_DEPS :=
COVERAGE_KERNEL_TARGETS :=
MEMORY_LAYOUT_LD_SCRIPT := $(LINKER_SCRIPTS)/memory.$(ARCH).ld
SFPI_LIBS :=
COVERAGE_BUILD := $(if $(coverage_build),$(coverage_build),false)
ifeq ($(COVERAGE_BUILD),true)
	SFPI_LIBS := -lgcov
	COVERAGE_DEPS :=  $(SHARED_OBJ_DIR)/coverage.o
	COVERAGE_KERNEL_TARGETS := $(OBJ_DIR)/kernel_%.gcno
	MEMORY_LAYOUT_LD_SCRIPT := $(LINKER_SCRIPTS)/memory.$(ARCH).debug.ld
	OPTIONS_COMPILE += -fprofile-arcs -ftest-coverage -fprofile-info-section -DCOVERAGE
endif

OPTIONS_LINK	:= -nodefaultlibs -fexceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -Wl,--trace

INCLUDES := -I../$(ARCH_LLK_ROOT)/llk_lib -I../$(ARCH_LLK_ROOT)/common/inc \
			-I../$(ARCH_LLK_ROOT)/common/inc/sfpu -Isfpi/compiler/lib/gcc/riscv-tt-elf/*/include -I$(HEADER_DIR) -Ifirmware/riscv/common \
			-Isfpi/include -Ihelpers/include -I$(TEST_DIR)

OPTIONS_COMPILE += $(INCLUDES)

TO_UPPER = $(shell echo $(1) | tr '[:lower:]' '[:upper:]')

# =========================
# Targets
# =========================
.PHONY: all dis profiler clean always

ELF_TARGETS := $(ELF_DIR)/unpack.elf \
				$(ELF_DIR)/math.elf \
				$(ELF_DIR)/pack.elf

DIS_TARGETS := $(DIS_DIR)/unpack.S \
				$(DIS_DIR)/math.S \
				$(DIS_DIR)/pack.S

ifneq ($(ARCH),quasar)
ELF_TARGETS += $(SHARED_ELF_DIR)/brisc.elf

DIS_TARGETS += $(SHARED_DIS_DIR)/brisc.S
endif

all: $(ELF_TARGETS)

dis: $(DIS_TARGETS)

profiler: $(PROFILER_DIR)/unpack.meta.bin \
	  $(PROFILER_DIR)/math.meta.bin \
	  $(PROFILER_DIR)/pack.meta.bin

# =========================
# Build Rules
# =========================


# extract profiler metadata from .elf
$(PROFILER_DIR)/%.meta.bin: $(ELF_DIR)/%.elf | $(PROFILER_DIR)
	$(OBJCOPY) -O binary -j .profiler_meta $< $@

# disassemble unpack.elf, math.elf, pack.elf
$(DIS_DIR)/%.S: $(ELF_DIR)/%.elf | $(DIS_DIR)
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

# disassemble brisc.elf
$(SHARED_DIS_DIR)/%.S: $(SHARED_ELF_DIR)/%.elf | $(SHARED_DIS_DIR)
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

# link unpack.elf, math.elf, pack.elf
$(ELF_DIR)/%.elf: $(SHARED_OBJ_DIR)/tmu-crt0.o $(OBJ_DIR)/main_%.o $(OBJ_DIR)/kernel_%.o $(COVERAGE_DEPS) | $(ELF_DIR) $(COVERAGE_REPORT_DIR)
	$(GXX) $(ARCH_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(MEMORY_LAYOUT_LD_SCRIPT) -T$(LINKER_SCRIPTS)/$*.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@ $(SFPI_LIBS)

# link brisc.elf
$(SHARED_ELF_DIR)/brisc.elf: $(SHARED_OBJ_DIR)/tmu-crt0.o $(SHARED_OBJ_DIR)/brisc.o $(COVERAGE_DEPS) | $(SHARED_ELF_DIR) $(COVERAGE_REPORT_DIR)
	$(GXX) $(ARCH_NON_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(MEMORY_LAYOUT_LD_SCRIPT) -T$(LINKER_SCRIPTS)/brisc.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@ $(SFPI_LIBS)

# disable make automatically deleting important intermediate files
.SECONDARY: $(SHARED_OBJ_DIR)/*.o $(OBJ_DIR)/*.o

# build kernel_unpack.o, kernel_math.o, kernel_pack.o from sources directory (checks ai_gen, quasar, then main)
# Find which source file exists and determine if COMPILE_FOR_TRISC flag is needed
KERNEL_SOURCE = $(firstword $(wildcard sources/ai_gen/$(testname).cpp sources/quasar/$(testname).cpp sources/$(testname).cpp))
KERNEL_TRISC_FLAG = $(if $(filter sources/quasar/%,$(KERNEL_SOURCE)),,-DCOMPILE_FOR_TRISC= )

$(OBJ_DIR)/kernel_%.o $(COVERAGE_KERNEL_TARGETS): $(KERNEL_SOURCE) $(BUILD_DIR)/params.stamp | $(OBJ_DIR) $(DEPS_DIR) $(TEST_DIR)/build.h
	$(GXX) $(ARCH_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(KERNEL_TRISC_FLAG) \
	-MMD -MP -MF $(patsubst $(OBJ_DIR)/%.o,$(DEPS_DIR)/%.d,$@) \
	-DLLK_TRISC_$(call TO_UPPER, $*) -c -o $@ $(KERNEL_SOURCE)

# build main_unpack.o, main_math.o, main_pack.o
$(OBJ_DIR)/main_%.o: $(RISCV_SOURCES)/trisc.cpp $(BUILD_DIR)/params.stamp | $(SHARED_DEPS_DIR)
	$(GXX) $(ARCH_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DCOMPILE_FOR_TRISC= \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-DLLK_TRISC_$(call TO_UPPER, $*) -c -o $@ $<

# build brisc.o
$(SHARED_OBJ_DIR)/brisc.o: $(RISCV_SOURCES)/brisc.cpp $(BUILD_DIR)/params.stamp | $(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR)
	$(GXX) $(ARCH_NON_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_COMPILE) \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-c -o $@ $<

$(SHARED_OBJ_DIR)/coverage.o: $(HELPERS)/src/coverage.cpp | $(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR)
	$(GXX) $(ARCH_NON_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -fno-strict-aliasing \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-c -o $@ $<

# build c runtime library
$(SHARED_OBJ_DIR)/tmu-crt0.o: $(HELPERS)/tmu-crt0.S | $(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR)
	$(GXX) $(ARCH_NON_COMPUTE) $(OPTIONS_ALL) $(OPTIONS_COMPILE) \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-c -o $@ $<

# create a stamp file to track build parameters
PARAMS := BOOT_MODE=$(BOOT_MODE) PROFILER_BUILD=$(PROFILER_BUILD)
$(BUILD_DIR)/params.stamp: always | $(BUILD_DIR)
	echo "$(PARAMS)" | cmp -s - $@ || echo "$(PARAMS)" > $@

# create folder structure root for stamp file
$(BUILD_DIR):
	mkdir -p $@

# create folder structure for shared files
$(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR) $(SHARED_DIS_DIR) $(SHARED_ELF_DIR) $(INFO_FILES_DIR):
	mkdir -p $@

# create folder structure for test specific files
$(OBJ_DIR) $(DEPS_DIR) $(DIS_DIR) $(ELF_DIR) $(PROFILER_DIR) $(COVERAGE_REPORT_DIR):
	mkdir -p $@


STREAM_FILES	:= $(wildcard $(TEST_DIR)/*.raw.stream)
coverage: $(INFO_FILES_DIR)
	@for f in $(STREAM_FILES); do \
        $(GCOV_TOOL) merge-stream "$$f"; \
    done

	lcov --gcov-tool $(GCOV) --capture \
	--directory $(OBJ_DIR) \
	--directory $(SHARED_OBJ_DIR) \
	--output-file $(INFO_FILES_DIR)/$(info_file_name) --rc lcov_branch_coverage=1

# =========================
# Clean
# =========================
clean:
	$(RMDIR) $(BUILD_DIR)
	$(RMDIR) __pycache__
	$(RMDIR) .pytest_cache
	$(MAKE) -C python_tests clean

# do nothing except cause dependent targets to always fire
always:

-include $(wildcard $(SHARED_DEPS_DIR)/*.d)
-include $(wildcard $(DEPS_DIR)/*.d)
