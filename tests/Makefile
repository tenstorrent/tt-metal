# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# =========================
# Toolchain and Directories
# =========================
CXX_VERSION     := c++17
TOOL_PATH       ?= sfpi/compiler/bin

BUILD_DIR       ?= build
SHARED_DIR       := $(BUILD_DIR)/shared
TEST_DIR        := $(BUILD_DIR)/tests/$(testname)
OBJ_DIR         := $(TEST_DIR)/obj
DIS_DIR         := $(TEST_DIR)/dis
PROFILER_DIR    := $(TEST_DIR)/profiler
ELF_DIR         := $(TEST_DIR)/elf

HELPERS         := helpers
RISCV_SOURCES   := $(HELPERS)/src
LINKER_SCRIPTS  := $(HELPERS)/ld
HEADER_DIR      := hw_specific/inc
RMDIR           := rm -rf

GXX             := $(TOOL_PATH)/riscv32-tt-elf-g++
OBJDUMP         := $(TOOL_PATH)/riscv32-tt-elf-objdump
OBJCOPY         := $(TOOL_PATH)/riscv32-tt-elf-objcopy

# =========================
# Architecture Selection
# =========================
ifeq ($(CHIP_ARCH),wormhole)
	ARCH_CPU	:= -mcpu=tt-wh
	ARCH_DEFINE     := -DARCH_WORMHOLE
	ARCH_LLK_ROOT   := tt_llk_wormhole_b0
else ifeq ($(CHIP_ARCH),blackhole)
	ARCH_CPU	:= -mcpu=tt-bh
	ARCH_DEFINE     := -DARCH_BLACKHOLE
	ARCH_LLK_ROOT   := tt_llk_blackhole
else
	$(error CHIP_ARCH must be either 'wormhole' or 'blackhole')
endif

# =========================
# Compiler and Linker Flags
# =========================
OPTIONS_ALL	:= -g -O3 -mabi=ilp32 -std=$(CXX_VERSION) -ffast-math $(ARCH_CPU)
OPTIONS_COMPILE := -fno-use-cxa-atexit -Wall -fpermissive -fno-exceptions -fno-rtti -Werror \
				   -Wno-unknown-pragmas -Wno-error=multistatement-macros -Wno-error=parentheses \
				   -Wno-error=unused-but-set-variable -Wno-unused-variable -DTENSIX_FIRMWARE \
				   $(ARCH_DEFINE) -DCOMPILE_FOR_TRISC=
OPTIONS_LINK	:= -fexceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -Wl,--trace

INCLUDES := -I../$(ARCH_LLK_ROOT)/llk_lib -I../$(ARCH_LLK_ROOT)/common/inc \
			-I../$(ARCH_LLK_ROOT)/common/inc/sfpu -I$(HEADER_DIR) \
			-Ifirmware/riscv/common -Ifirmware/riscv/$(CHIP_ARCH) \
			-Isfpi/include -Ihelpers/include

OPTIONS_COMPILE += $(INCLUDES)

TO_UPPER = $(shell echo $(1) | tr '[:lower:]' '[:upper:]')

# =========================
# Targets
# =========================
.PHONY: all dis profiler clean

all: $(ELF_DIR)/unpack.elf \
	 $(ELF_DIR)/math.elf \
	 $(ELF_DIR)/pack.elf \
	 $(SHARED_DIR)/brisc.elf

dis: $(DIS_DIR)/unpack.S \
	 $(DIS_DIR)/math.S \
	 $(DIS_DIR)/pack.S \
	 $(SHARED_DIR)/brisc.S

profiler: $(PROFILER_DIR)/unpack.meta.bin \
	  $(PROFILER_DIR)/math.meta.bin \
	  $(PROFILER_DIR)/pack.meta.bin
# =========================
# Build Rules
# =========================

# Currently only used for brisc.dis
$(SHARED_DIR)/%.S: $(SHARED_DIR)/%.elf | $(SHARED_DIR)
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

$(PROFILER_DIR)/%.meta.bin: $(ELF_DIR)/%.elf | $(PROFILER_DIR)
	$(OBJCOPY) -O binary -j .profiler_meta $< $@

$(DIS_DIR)/%.S: $(ELF_DIR)/%.elf | $(DIS_DIR)
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

$(ELF_DIR)/%.elf: $(SHARED_DIR)/tmu-crt0.o $(SHARED_DIR)/main_%.o $(OBJ_DIR)/kernel_%.o | $(ELF_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(CHIP_ARCH).ld -T$(LINKER_SCRIPTS)/$*.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@

# Building brisc.elf for BRISC core
$(SHARED_DIR)/brisc.elf: $(SHARED_DIR)/tmu-crt0.o $(SHARED_DIR)/brisc.o | $(SHARED_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(CHIP_ARCH).ld -T$(LINKER_SCRIPTS)/brisc.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@

$(OBJ_DIR)/kernel_%.o: sources/$(testname).cpp | $(OBJ_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -MMD -MP -MF $(@:.o=.d) -DLLK_TRISC_$(call TO_UPPER, $*) -c -o $@ $<

# compiling main for every TRISC core
$(SHARED_DIR)/main_%.o: $(RISCV_SOURCES)/trisc.cpp | $(SHARED_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -MMD -MP -MF $(@:.o=.d) -DLLK_TRISC_$(call TO_UPPER, $*) -c -o $@ $<

$(SHARED_DIR)/brisc.o: $(RISCV_SOURCES)/brisc.cpp | $(SHARED_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

$(SHARED_DIR)/tmu-crt0.o: $(HELPERS)/tmu-crt0.S | $(SHARED_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

# required folder structure
$(PROFILER_DIR) $(DIS_DIR) $(ELF_DIR) $(OBJ_DIR) $(SHARED_DIR):
	mkdir -p $@

# =========================
# Clean
# =========================
clean:
	$(RMDIR) $(BUILD_DIR)
	$(RMDIR) __pycache__
	$(RMDIR) .pytest_cache
	$(MAKE) -C python_tests clean

-include $(wildcard $(OBJ_DIR)/*.d)
-include $(wildcard $(SHARED_DIR)/*.d)
