# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# =========================
# Toolchain and Directories
# =========================
CXX_VERSION     := c++17
TOOL_PATH       ?= sfpi/compiler/bin
BUILD_DIR       ?= build
OUTPUT_ELFS     := $(BUILD_DIR)/elf
HELPERS         := helpers
RISCV_SOURCES   := $(HELPERS)/src
LINKER_SCRIPTS  := $(HELPERS)/ld
HEADER_DIR      := hw_specific/inc
RMDIR           := rm -rf

GXX             := $(TOOL_PATH)/riscv32-tt-elf-g++
OBJDUMP         := $(TOOL_PATH)/riscv32-tt-elf-objdump

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
OPTIONS_ALL	:= -O3 -mabi=ilp32 -std=$(CXX_VERSION) -ffast-math $(ARCH_CPU)
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

# =========================
# Targets
# =========================
.PHONY: all dis clean

all: $(BUILD_DIR) $(OUTPUT_ELFS) \
	 $(OUTPUT_ELFS)/$(testname)_trisc0.elf \
	 $(OUTPUT_ELFS)/$(testname)_trisc1.elf \
	 $(OUTPUT_ELFS)/$(testname)_trisc2.elf \
	 $(OUTPUT_ELFS)/brisc.elf

dis: $(BUILD_DIR) $(OUTPUT_ELFS) \
	 $(BUILD_DIR)/$(testname)_trisc0.dis \
	 $(BUILD_DIR)/$(testname)_trisc1.dis \
	 $(BUILD_DIR)/$(testname)_trisc2.dis \
	 $(BUILD_DIR)/brisc.dis

# =========================
# Build Rules
# =========================

$(BUILD_DIR)/%.dis :  $(OUTPUT_ELFS)/%.elf
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

$(BUILD_DIR):
	mkdir -p $@

$(OUTPUT_ELFS): $(BUILD_DIR)
	mkdir -p $@

# Explicit rules for TRISC .elfs
# Buiding .elf files for every TRISC core
$(OUTPUT_ELFS)/$(testname)_trisc0.elf :$(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/main_unpack.o $(BUILD_DIR)/$(testname)_unpack.o
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(CHIP_ARCH).ld -T$(LINKER_SCRIPTS)/trisc0.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@
$(OUTPUT_ELFS)/$(testname)_trisc1.elf: $(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/main_math.o $(BUILD_DIR)/$(testname)_math.o
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(CHIP_ARCH).ld -T$(LINKER_SCRIPTS)/trisc1.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@
$(OUTPUT_ELFS)/$(testname)_trisc2.elf: $(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/main_pack.o $(BUILD_DIR)/$(testname)_pack.o
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(CHIP_ARCH).ld -T$(LINKER_SCRIPTS)/trisc2.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@

# Building brisc.elf for BRISC core
$(OUTPUT_ELFS)/brisc.elf: $(BUILD_DIR)/tmu-crt0.o $(BUILD_DIR)/brisc.o
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_LINK) $^ -T$(LINKER_SCRIPTS)/memory.$(CHIP_ARCH).ld -T$(LINKER_SCRIPTS)/brisc.ld -T$(LINKER_SCRIPTS)/sections.ld -o $@

# force recompilation of tests per core
# force recompilation of trisc.cpp for ZONE_SCOPED("KERNEL") to work
.PHONY: $(BUILD_DIR)/$(testname)_unpack.o $(BUILD_DIR)/$(testname)_math.o $(BUILD_DIR)/$(testname)_pack.o \
        $(BUILD_DIR)/main_unpack.o $(BUILD_DIR)/main_math.o $(BUILD_DIR)/main_pack.o

$(BUILD_DIR)/$(testname)_unpack.o: sources/$(testname).cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/$(testname)_math.o: sources/$(testname).cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/$(testname)_pack.o: sources/$(testname).cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DLLK_TRISC_PACK -c -o $@ $<

#compiling main for every TRISC core
$(BUILD_DIR)/main_unpack.o : $(RISCV_SOURCES)/trisc.cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/main_math.o : $(RISCV_SOURCES)/trisc.cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/main_pack.o : $(RISCV_SOURCES)/trisc.cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -DLLK_TRISC_PACK -c -o $@ $<

$(BUILD_DIR)/tmu-crt0.o: $(HELPERS)/tmu-crt0.S | $(BUILD_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -c -o $@ $<

$(BUILD_DIR)/brisc.o: $(RISCV_SOURCES)/brisc.cpp | $(BUILD_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -c -o $@ $<

# =========================
# Clean
# =========================
clean:
	$(RMDIR) $(BUILD_DIR)
	$(RMDIR) __pycache__
	$(RMDIR) .pytest_cache
	$(MAKE) -C python_tests clean
