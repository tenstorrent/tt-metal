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
	ARCH_CPU        := -mcpu=tt-wh
	ARCH_DEFINE     := -DARCH_WORMHOLE
	ARCH_LLK_ROOT   := tt_llk_wormhole_b0
else ifeq ($(ARCH),blackhole)
	ARCH_CPU        := -mcpu=tt-bh
	ARCH_DEFINE     := -DARCH_BLACKHOLE
	ARCH_LLK_ROOT   := tt_llk_blackhole
else
$(error must provide archname or CHIP_ARCH environment variable (wormhole / blackhole))
endif

BUILD_DIR       ?= build/$(ARCH)

SHARED_DIR      := $(BUILD_DIR)/shared
SHARED_OBJ_DIR  := $(SHARED_DIR)/obj
SHARED_DEPS_DIR := $(SHARED_DIR)/deps
SHARED_DIS_DIR  := $(SHARED_DIR)/dis
SHARED_ELF_DIR  := $(SHARED_DIR)/elf

TEST_DIR        := $(BUILD_DIR)/tests/$(testname)
OBJ_DIR         := $(TEST_DIR)/obj
DEPS_DIR        := $(TEST_DIR)/deps
DIS_DIR         := $(TEST_DIR)/dis
ELF_DIR         := $(TEST_DIR)/elf

PROFILER_DIR    := $(TEST_DIR)/profiler

HELPERS         := helpers
RISCV_SOURCES   := $(HELPERS)/src
LINKER_SCRIPTS  := $(HELPERS)/ld
HEADER_DIR      := hw_specific/$(ARCH)/inc
RMDIR           := rm -rf

GXX             := $(TOOL_PATH)/riscv32-tt-elf-g++
OBJDUMP         := $(TOOL_PATH)/riscv32-tt-elf-objdump
OBJCOPY         := $(TOOL_PATH)/riscv32-tt-elf-objcopy

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
			-Ifirmware/riscv/common -Ifirmware/riscv/$(ARCH) \
			-Isfpi/include -Ihelpers/include

OPTIONS_COMPILE += $(INCLUDES)

TO_UPPER = $(shell echo $(1) | tr '[:lower:]' '[:upper:]')

TEST_KERNEL_FLAG := -DTEST_KERNEL

# =========================
# Targets
# =========================
.PHONY: all dis profiler clean

all: $(ELF_DIR)/unpack.elf \
	 $(ELF_DIR)/math.elf \
	 $(ELF_DIR)/pack.elf \
	 $(SHARED_ELF_DIR)/brisc.elf

dis: $(DIS_DIR)/unpack.S \
	 $(DIS_DIR)/math.S \
	 $(DIS_DIR)/pack.S \
	 $(SHARED_DIS_DIR)/brisc.S

profiler: $(PROFILER_DIR)/unpack.meta.bin \
	  $(PROFILER_DIR)/math.meta.bin \
	  $(PROFILER_DIR)/pack.meta.bin


# =========================
# Build Rules
# =========================


# extract profiler metadata from .elf
$(PROFILER_DIR)/%.meta.bin: $(ELF_DIR)/%.elf | $(PROFILER_DIR)
	$(OBJCOPY) -O binary -j .profiler_meta $< $@

# disasseble unpack.elf, math.elf, pack.elf
$(DIS_DIR)/%.S: $(ELF_DIR)/%.elf | $(DIS_DIR)
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

# disassemble brisc.elf
$(SHARED_DIS_DIR)/%.S: $(SHARED_ELF_DIR)/%.elf | $(SHARED_DIS_DIR)
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

# link unpack.elf, math.elf, pack.elf
$(ELF_DIR)/%.elf: $(SHARED_OBJ_DIR)/tmu-crt0.o $(SHARED_OBJ_DIR)/main_%.o $(OBJ_DIR)/kernel_%.o | $(ELF_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(ARCH).ld -T$(LINKER_SCRIPTS)/$*.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@

# link brisc.elf
$(SHARED_ELF_DIR)/brisc.elf: $(SHARED_OBJ_DIR)/tmu-crt0.o $(SHARED_OBJ_DIR)/brisc.o | $(SHARED_ELF_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(ARCH).ld -T$(LINKER_SCRIPTS)/brisc.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@

# disable make automatically deleting important intermediate files
.SECONDARY: $(SHARED_OBJ_DIR)/*.o $(OBJ_DIR)/*.o

# build kernel_unpack.o, kernel_math.o, kernel_pack.o
$(OBJ_DIR)/kernel_%.o: sources/$(testname).cpp | $(OBJ_DIR) $(DEPS_DIR)
	$(GXX) $(OPTIONS_ALL) $(TEST_KERNEL_FLAG) $(OPTIONS_COMPILE) \
	-MMD -MP -MF $(patsubst $(OBJ_DIR)/%.o,$(DEPS_DIR)/%.d,$@) \
	-DLLK_TRISC_$(call TO_UPPER, $*) -c -o $@ $<

# build main_unpack.o, main_math.o, main_pack.o
$(SHARED_OBJ_DIR)/main_%.o: $(RISCV_SOURCES)/trisc.cpp | $(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-DLLK_TRISC_$(call TO_UPPER, $*) -c -o $@ $<

# build brisc.o
$(SHARED_OBJ_DIR)/brisc.o: $(RISCV_SOURCES)/brisc.cpp | $(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-c -o $@ $<

# build c runtime library
$(SHARED_OBJ_DIR)/tmu-crt0.o: $(HELPERS)/tmu-crt0.S | $(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) \
	-MMD -MP -MF $(patsubst $(SHARED_OBJ_DIR)/%.o,$(SHARED_DEPS_DIR)/%.d,$@) \
	-c -o $@ $<

# create folder structure for shared files
$(SHARED_OBJ_DIR) $(SHARED_DEPS_DIR) $(SHARED_DIS_DIR) $(SHARED_ELF_DIR):
	mkdir -p $@

# create folder structure for test specific files
$(OBJ_DIR) $(DEPS_DIR) $(DIS_DIR) $(ELF_DIR) $(PROFILER_DIR):
	mkdir -p $@


# =========================
# Clean
# =========================
clean:
	$(RMDIR) build
	$(RMDIR) __pycache__
	$(RMDIR) .pytest_cache
	$(MAKE) -C python_tests clean

-include $(wildcard $(SHARED_DEPS_DIR)/*.d)
-include $(wildcard $(DEPS_DIR)/*.d)
