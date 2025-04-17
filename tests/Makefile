# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

TOOL_PATH?=sfpi/compiler/bin
BUILD_DIR ?=build

RMDIR := rm -rf

# Tool paths, mfc changing from 'unknown' to 'tt'
GXX:=$(wildcard $(TOOL_PATH)/riscv32-*-elf-g++)
OBJDUMP:=$(wildcard $(TOOL_PATH)/riscv32-*-elf-objdump)

# GCC options
OPTIONS_ALL=-O3 -mabi=ilp32 -std=c++20 -ffast-math -ftest-coverage
OPTIONS_COMPILE=-fno-use-cxa-atexit -Wall -fpermissive -fno-exceptions -fno-rtti -Werror -Wno-unknown-pragmas -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -DTENSIX_FIRMWARE #-DDEBUG_PRINT_ENABLED

# Default values
ARCH_ROOT := unknown

# Architecture-specific options
ifeq ($(CHIP_ARCH), wormhole)
    ARCH_OPTS := -mcpu=tt-wh -DARCH_WORMHOLE
    ARCH_LLK_ROOT := tt_llk_wormhole_b0
else ifeq ($(CHIP_ARCH), blackhole)
    ARCH_OPTS := -mcpu=tt-bh -DARCH_BLACKHOLE
    ARCH_LLK_ROOT := tt_llk_blackhole
else
    $(error CHIP_ARCH is neither wormhole nor blackhole.)
	exit (1)
endif

# Append architecture-specific options
OPTIONS_ALL += $(filter -mcpu=%, $(ARCH_OPTS))
OPTIONS_COMPILE += $(filter -DARCH_%, $(ARCH_OPTS))

OPTIONS_LINK=-fexceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -Wl,--trace
INCLUDES = -I../$(ARCH_LLK_ROOT)/llk_lib -I../$(ARCH_LLK_ROOT)/common/inc -I../$(ARCH_LLK_ROOT)/common/inc/sfpu -I../$(ARCH_LLK_ROOT)/hw_specific/inc
INCLUDES += -Ifirmware/riscv/common -Ifirmware/riscv/$(CHIP_ARCH)/ -Isfpi/include -Ihelpers/include

FORMAT_ARG := -D$(unpack_A_src) -D$(unpack_A_dst) -D$(unpack_B_src) -D$(unpack_B_dst) -D$(fpu) -D$(pack_src) -D$(pack_dst)

ifeq ($(mathop),)
    MATHOP_ARG =
else
	MATHOP_ARG:= -D$(mathop) -DAPPROX_MODE=$(approx_mode)  -DPOOL_TYPE=$(pool_type) -DREDUCE_DIM=$(reduce_dim)
endif

OPTIONS_COMPILE+= -DMATH_FIDELITY=$(math_fidelity)

# If test needs to run muliple consecutive operations it needs some additional
# files to be linked to it
ifeq ($(testname), multiple_tiles_eltwise_test)
	MULTIPLE_OPS:=-DMULTIPLE_OPS
	OPTIONS_UNPACK:=-DKERN_CNT=$(kern_cnt)
	OPTIONS_MATH:=-DKERN_CNT=$(kern_cnt)
	OPTIONS_PACK:=-DKERN_CNT=$(kern_cnt)
	OPTIONS_PACK+=-DPACK_ADDR_CNT=$(pack_addr_cnt)
	OPTIONS_PACK+=-DPACK_ADDRS=$(pack_addrs)
endif

OPTIONS_COMPILE+=$(FORMAT_ARG) $(MATHOP_ARG) -DTILE_SIZE_CNT=0x1000
OPTIONS_COMPILE+=$(INCLUDES)

ifeq ($(dest_acc), DEST_ACC)
	OPTIONS_COMPILE+=-DDEST_ACC
endif

# Define project paths
HELPERS=helpers
RISCV_SOURCES=$(HELPERS)/src
LINKER_SCRIPTS=$(HELPERS)/ld
OUTPUT_ELFS = $(BUILD_DIR)/elf

# Define targets
.PHONY: all clean

all: $(BUILD_DIR) $(OUTPUT_ELFS) $(OUTPUT_ELFS)/$(testname)_trisc0.elf $(OUTPUT_ELFS)/$(testname)_trisc1.elf $(OUTPUT_ELFS)/$(testname)_trisc2.elf $(OUTPUT_ELFS)/brisc.elf
dis: $(BUILD_DIR) $(OUTPUT_ELFS) $(BUILD_DIR)/$(testname)_trisc0.dis $(BUILD_DIR)/$(testname)_trisc1.dis $(BUILD_DIR)/$(testname)_trisc2.dis $(BUILD_DIR)/brisc.dis

$(BUILD_DIR)/%.dis :  $(OUTPUT_ELFS)/%.elf
	$(OBJDUMP) -xsD $< > $@
	$(OBJDUMP) -t $< | sort >> $@

$(BUILD_DIR):
	mkdir -p $@

$(OUTPUT_ELFS): $(BUILD_DIR)
	mkdir -p $@

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

#compiling _test.pp to .o
$(BUILD_DIR)/$(testname)_unpack.o: sources/$(testname).cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_UNPACK) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/$(testname)_math.o: sources/$(testname).cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_MATH) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/$(testname)_pack.o: sources/$(testname).cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_PACK) -DLLK_TRISC_PACK -c -o $@ $<

#compiling main for every TRISC core
$(BUILD_DIR)/main_unpack.o : $(RISCV_SOURCES)/trisc.cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_UNPACK) $(MULTIPLE_OPS) -DLLK_TRISC_UNPACK -c -o $@ $<
$(BUILD_DIR)/main_math.o : $(RISCV_SOURCES)/trisc.cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_MATH) $(MULTIPLE_OPS) -DLLK_TRISC_MATH -c -o $@ $<
$(BUILD_DIR)/main_pack.o : $(RISCV_SOURCES)/trisc.cpp
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) $(OPTIONS_PACK) $(MULTIPLE_OPS) -DLLK_TRISC_PACK -c -o $@ $<

$(BUILD_DIR)/tmu-crt0.o: $(HELPERS)/tmu-crt0.S | $(BUILD_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -c -o $@ $<

$(BUILD_DIR)/brisc.o: $(RISCV_SOURCES)/brisc.cpp | $(BUILD_DIR)
	$(GXX) $(OPTIONS_ALL) $(OPTIONS_COMPILE) -c -o $@ $<

clean:
	$(RMDIR) $(BUILD_DIR)
	$(RMDIR) __pycache__
	$(RMDIR) .pytest_cache
	$(MAKE) -C python_tests clean
