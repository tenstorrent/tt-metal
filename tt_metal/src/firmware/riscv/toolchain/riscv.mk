# 1. Set SOURCES to your list of C/C++ sources (default main.cpp)
# 2. Override any other variables listed below.
# 3. Include this file.
# 4. Extend any targets as needed (all, extras, clean), see below.
#
# Output will be produced in $(OUTPUT_DIR)/$(FIRMWARE_NAME).hex (also .elf, .map).
#
# Variables you can override:
# - SOURCES: list of source files, paths relative to the Makefile, cpp, cc, c, S supported (main.cpp)
# - OPT_FLAGS: optimisation flags for C, C++ and C/C++ link (-flto -Os -g)
# - C_LANG_FLAGS: C language dialect & diagnostics (-Wall)
# - CXX_LANG_FLAGS: C++ language dialect & diagnostics (-Wall -std=c++14)
# - DEFINES: Preprocessor definitions for C, C++ and assembly ()
# - INCLUDES: additional include directories other than your source directories
# - OUTPUT_DIR: subdirectory for all outputs and temporary files, will be created if necessary (out)
# - FIRMWARE_NAME: firmware file name (firmware)
# - INFO_NAME: basename for ancillary files such as fwlog and debug info
# - FIRMWARE_START: start address of firmware (includes magic variables such as TEST_MAILBOX) (0)
#
# Targets you can extend: all, extras, clean. Use "::" instead of ":".

all:: # Always first to guarantee all is the default goal.

TOOLCHAIN := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

include $(TT_METAL_HOME)/tt_metal/common/common.mk

SFPI ?= $(TT_METAL_HOME)/tt_metal/src/ckernels/sfpi
RISCV_TOOLS_PREFIX := $(SFPI)/compiler/bin/riscv32-unknown-elf-
# RISCV_TOOLS_PREFIX := /home/software/risc-v/riscv64-unknown-elf-gcc-8.3.0-2020.04.0-x86_64-linux-ubuntu14/bin/riscv64-unknown-elf-
CXX := $(CCACHE) $(RISCV_TOOLS_PREFIX)g++
CC  := $(CCACHE) $(RISCV_TOOLS_PREFIX)gcc
AS  := $(CCACHE) $(RISCV_TOOLS_PREFIX)gcc
OBJDUMP := $(RISCV_TOOLS_PREFIX)objdump
OBJCOPY := $(RISCV_TOOLS_PREFIX)objcopy

TDMA_ASSEMBLER_DIR := $(TT_METAL_HOME)/src/software/assembler
TDMA_ASSEMBLER := $(TDMA_ASSEMBLER_DIR)/out/assembler

DEP_FLAGS := -MD -MP
OPT_FLAGS ?= -flto -ffast-math -O3 -g
# OPT_FLAGS ?= -flto -O2 -g
C_LANG_FLAGS ?= -Wall -Werror
# -fno-use-cax-atexit tells the compiler to use regular atexit for global object cleanup.
# Even for RISC-V, GCC compatible compilers use the Itanium ABI for some c++ things.
# atexit vs __cxa_atexit: https://itanium-cxx-abi.github.io/cxx-abi/abi.html#dso-dtor-runtime-api
#CXX_LANG_FLAGS ?= -Wall -Werror -std=c++17 -Wno-unknown-pragmas -fno-use-cxa-atexit -Wno-error=multistatement-macros
CXX_LANG_FLAGS ?= -Wall -Werror -std=c++17 -Wno-unknown-pragmas -fno-use-cxa-atexit -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -fno-exceptions
DEFINES ?=
SOURCES ?= main.cpp

DISABLE_FWLOG ?= 1
CKDEBUG ?= 0

ifeq ($(DISABLE_FWLOG),0)
	DEFINES += -D ENABLE_FWLOG
endif

ifeq ($(CKDEBUG),1)
	DEFINES += -D ENABLE_FWLOG
endif

ifeq ($(FW_GPR_ANNOTATION),1)
	DEFINES += -D FW_GPR_ANNOTATION
endif

ifeq ($(NO_DISTRIBUTED_EPOCH_TABLES),1)
	DEFINES += $(addprefix -DNO_DISTRIBUTED_EPOCH_TABLES=, $(NO_DISTRIBUTED_EPOCH_TABLES))
endif


ifeq ($(ARCH_NAME),grayskull)
ARCH_FLAG := "-mgrayskull"
else ifeq ($(ARCH_NAME),wormhole_b0)
ARCH_FLAG := "-mwormhole"
endif
TRISC_L0_EN ?= 0
ARCH := -march=rv32i -mabi=ilp32 $(ARCH_FLAG)
# ARCH := -march=rv32i -mabi=ilp32
LINKER_SCRIPT_NAME ?= tensix.ld
LINKER_SCRIPT_SRC := $(TOOLCHAIN)/$(LINKER_SCRIPT_NAME)
DEFS := -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=$(TRISC_L0_EN)

ARCH_NAME ?= grayskull

OUTPUT_DIR ?= $(TT_METAL_HOME)/build/src/firmware/riscv/targets/$(FIRMWARE_NAME)/out
LINKER_SCRIPT := $(OUTPUT_DIR)/$(LINKER_SCRIPT_NAME)
GEN_DIR := gen
FIRMWARE_NAME ?= firmware
INFO_NAME ?= $(FIRMWARE_NAME)
FIRMWARE_START ?= 0

# All objects are dumped into out, so we don't support two source files in different directories with the same name.
ifneq ($(words $(sort $(SOURCES))),$(words $(sort $(notdir $(SOURCES)))))
$(error $$(SOURCES) contains a duplicate filename)
endif

# Derive the list of source directories from $(SOURCES), use that as a list of include directories.
SOURCE_DIRS := $(filter-out ./,$(sort $(dir $(SOURCES))))
# rk: TODO: get rid of one of TT_METAL_HOME
INCLUDES := $(INCLUDES) -I "$(TT_METAL_HOME)" -I "$(TT_METAL_HOME)/tt_metal" -I "$(SFPI)/include" -I "$(TT_METAL_HOME)/src/firmware/riscv/common" $(addprefix -iquote ,$(SOURCE_DIRS)) -iquote .

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/wormhole"
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/wormhole/noc"
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/wormhole/wormhole_b0_defines"
else
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/grayskull/grayskull_defines"
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/grayskull"
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/grayskull/noc"
endif

ifeq ("$(ARCH_NAME)", "wormhole")
  INCLUDES += -I "$(TT_METAL_HOME)/src/firmware/riscv/wormhole/wormhole_a0_defines"
endif

# These are deferred so I can adjust DEP_FLAGS in the dependency-generation-only rules
CXXFLAGS = $(ARCH) $(DEP_FLAGS) $(OPT_FLAGS) $(CXX_LANG_FLAGS) $(DEFINES) $(DEFS) $(INCLUDES)
CFLAGS = $(ARCH) $(DEP_FLAGS) $(OPT_FLAGS) $(C_LANG_FLAGS) $(DEFINES) $(DEFS) $(INCLUDES)

LDFLAGS := $(ARCH) $(OPT_FLAGS) -Wl,--gc-sections -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -Wl,--defsym=__firmware_start=$(FIRMWARE_START) \
-T$(LINKER_SCRIPT) -L$(TOOLCHAIN) -nostartfiles

OUTFW := $(OUTPUT_DIR)/$(FIRMWARE_NAME)

OBJECTS := $(addprefix $(OUTPUT_DIR)/, $(addsuffix .o,$(basename $(notdir $(SOURCES)))))
DEPENDS := $(addprefix $(OUTPUT_DIR)/, $(addsuffix .d,$(basename $(notdir $(SOURCES)))))

ifneq ("$(BUILD_TARGET)", "ETH_APP")
  EXTRA_OBJECTS += $(OUTPUT_DIR)/substitutes.o $(OUTPUT_DIR)/tmu-crt0.o
endif

vpath % $(subst $(space),:,$(TOOLCHAIN) $(SOURCE_DIRS))

#stopping bin generation for now
all:: extras $(OUTFW).hex $(OUTFW).map #$(OUTFW).bin
	@$(PRINT_SUCCESS)

$(GEN_DIR):
	-mkdir -p $@

# These dependency-generation-only rules are here so we can rebuild dependencies
# without running the full compiler, and thereby discover generated header
# files (but not their dependencies).

# Special DEP_FLAGS to scan deps only, not compile. This variable will
# be applied whenever building *.d.
$(OUTPUT_DIR)/%.d: DEP_FLAGS = -M -MP -MG -MT $(addsuffix .o,$(basename $@))

# For C or C++ sources, regenerate dependencies by running the compiler with
# the special DEP_FLAGS above.

DEBUG_MODE ?= 0
POTENTIAL_DEBUG_FLAGS ?= $(CXXFLAGS)

# TODO(agrebenisan): Add ncrisc to this list
DEBUG_OPTIONS_SO_FAR := brisc # ncrisc
ifeq ($(DEBUG_MODE),1)
	ifneq ($(filter $(INFO_NAME), $(DEBUG_OPTIONS_SO_FAR)),)
		POTENTIAL_DEBUG_FLAGS = $(ARCH) $(DEP_FLAGS) -ffast-math -O1 -g -gdwarf-2 $(CXX_LANG_FLAGS) $(DEFINES) $(DEFS) $(INCLUDES)
	endif
endif

$(OUTPUT_DIR)/%.ld: %.ld | $(OUTPUT_DIR)
	$(CXX) $(DEFINES) $(DEFS) $(INCLUDES) -E -P -x c -o $@ $<

$(OUTPUT_DIR)/%.d: %.c | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -x c++ -c -o $@ $<

$(OUTPUT_DIR)/%.d: %.cpp | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OUTPUT_DIR)/%.d: %.cc | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# All objects depend on the dependency listing. Maybe this should be an order-only rule.
$(OUTPUT_DIR)/%.o: $(OUTPUT_DIR)/%.d

# Assemble tensix assembly into an array fragment
$(GEN_DIR)/%.asm.h: %.asm | $(GEN_DIR)
	@echo TDMA_AS $<
	$(TDMA_ASSEMBLER) --out-array $@ $<

$(OUTPUT_DIR):
	-mkdir -p $@

# Assemble RISC-V sources using C compiler
$(OUTPUT_DIR)/%.o: %.S | $(OUTPUT_DIR)
	@echo "AS $<"
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile C
$(OUTPUT_DIR)/%.o: %.c | $(OUTPUT_DIR)
	@echo "CC $<"
	$(CC) $(CXXFLAGS) -x c++ -c -o $@ $<

# Compile C++
$(OUTPUT_DIR)/%.o: %.cpp | $(OUTPUT_DIR)
	@echo "CXX $<"
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile C++
$(OUTPUT_DIR)/%.o: %.cc | $(OUTPUT_DIR)
	echo "$(CXX) $(CXXFLAGS) -c -o $@ $<"
	@echo "CXX $<"
	$(CXX) $(POTENTIAL_DEBUG_FLAGS) -c -o $@ $<
# $(OUTPUT_DIR)/%.o: %.cc | $(OUTPUT_DIR)
# 	echo "$(CXX) $(CXXFLAGS) -c -o $@ $<"
# 	@echo "CXX $<"
# 	$(CXX) $(CXXFLAGS) -c -o $@ $<


# Link using C++ compiler
$(OUTFW).elf: $(OBJECTS) $(EXTRA_OBJECTS) $(LINKER_SCRIPT)
	@echo "$(CKERNELS)"
	@echo "$(CXX) $(LDFLAGS) -o $@ $(filter-out $(LINKER_SCRIPT),$^)"
	$(CXX) $(LDFLAGS) -o $@ $(filter-out $(LINKER_SCRIPT),$^)
	chmod -x $@

# Generate hex-formatted firmware for LUA
$(OUTFW).hex: $(OUTFW).elf
	$(OBJCOPY) -O verilog $< $@.tmp
	python3 $(TOOLCHAIN)/hex8tohex32.py $@.tmp > $@
	rm $@.tmp

$(OUTFW).bin: $(OUTFW).elf
	echo $(OBJCOPY) -R .data -O binary $< $@
	$(OBJCOPY) -R .data -O binary $< $@

# Symbol map
$(OUTFW).map: $(OUTFW).elf
	$(OBJDUMP) -Stg $< > $@

# Create a file that maps where we log in firmware source to what is being logged
# IMPROVE: handle multiple source files
$(OUTPUT_DIR)/$(INFO_NAME).fwlog: $(OUTFW).elf
	python3 $(TT_METAL_HOME)/src/firmware/riscv/toolchain/fwlog.py --depfile $(OUTPUT_DIR)/$(INFO_NAME).d > $@

CKERNEL_DEPS := $(wildcard $(CKERNELS_DIR)/$(ARCH_NAME)/src/out/*.d)

$(OUTPUT_DIR)/ckernel.fwlog: $(OUTFW).elf $(CKERNEL_DEPS) $(OUTPUT_DIR)/ckernel.d
	python3 $(TT_METAL_HOME)/src/firmware/riscv/toolchain/fwlog.py --depfile $(OUTPUT_DIR)/ckernel.d > $@.tmp
	python3 $(TT_METAL_HOME)/src/firmware/riscv/toolchain/fwlog.py --depfile $(CKERNEL_DEPS) --path=$(CKERNELS_DIR)/$(ARCH_NAME)/src >> $@.tmp
#	common .cc files that are not in any dependenecies
	python3 $(TT_METAL_HOME)/src/firmware/riscv/toolchain/fwlog.py --depfile $(CKERNELS_DIR)/$(ARCH_NAME)/common/src/fwlog_list --path=$(CKERNELS_DIR)/$(ARCH_NAME)/common/src >> $@.tmp
	sort -u $@.tmp > $@   # uniquify
	rm -f $@.tmp

# Create a map between source files and the PC
$(OUTPUT_DIR)/$(INFO_NAME)-decodedline.txt: $(OUTFW).elf
	$(RISCV_TOOLS_PREFIX)readelf --debug-dump=decodedline $< > $@

# Create a map between source files and the PC
$(OUTPUT_DIR)/$(INFO_NAME)-debuginfo.txt: $(OUTFW).elf
	$(RISCV_TOOLS_PREFIX)readelf --debug-dump=info $< > $@
#	python ./parse_labels.py --infile out/firmware-debuginfo.txt

# Create a symbol table dump
$(OUTPUT_DIR)/$(INFO_NAME)-symbols.txt: $(OUTFW).elf
	$(RISCV_TOOLS_PREFIX)objdump -Stg $< > $@

clean2::
	rm $(OUTFW).elf $(OUTFW).hex $(OUTFW).bin $(OUTFW).map $(SILENT_ERRORS)
	rm $(OBJECTS) $(SILENT_ERRORS)
	rm $(DEPENDS) $(SILENT_ERRORS)
	rm $(EXTRA_OBJECTS) $(SILENT_ERRORS)
	rm $(EXTRA_OBJECTS:.o=.d) $(SILENT_ERRORS)
	rmdir $(OUTPUT_DIR) $(SILENT_ERRORS)
	rm $(OUTPUT_DIR)/$(INFO_NAME).fwlog $(OUTPUT_DIR)/$(INFO_NAME)-decodedline.txt $(OUTPUT_DIR)/$(INFO_NAME)-debuginfo.txt $(OUTPUT_DIR)/$(INFO_NAME)-symbols.txt $(SILENT_ERRORS)
	rm -rf $(GEN_DIR) $(SILENT_ERRORS)

extras::

.PHONY: clean2 all extras

-include $(DEPENDS)
