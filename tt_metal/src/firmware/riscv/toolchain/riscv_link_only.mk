all:: # Always first to guarantee all is the default goal.


SFPI ?= $(TT_METAL_HOME)/tt_metal/src/ckernels/sfpi

ifeq ($(RELEASE), 1)
TOOLCHAIN := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
RISCV_TOOLS_PREFIX := $(SFPI)/compiler/bin/riscv32-unknown-elf-
else
TOOLCHAIN := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
RISCV_TOOLS_PREFIX := $(SFPI)/compiler/bin/riscv32-unknown-elf-
endif


CKERNELS ?= $(CKERNELS_DIR)
CXX := $(CCACHE) $(RISCV_TOOLS_PREFIX)g++
OBJDUMP := $(RISCV_TOOLS_PREFIX)objdump
OBJCOPY := $(RISCV_TOOLS_PREFIX)objcopy
OPT_FLAGS ?= -flto -ffast-math -O3 -g -fno-exceptions

SOURCES ?= main.cpp

ifeq ($(ARCH_NAME),grayskull)
ARCH_FLAG := "-mgrayskull"
else ifeq ($(ARCH_NAME),wormhole_b0)
ARCH_FLAG := "-mwormhole"
endif
ARCH := -march=rv32i $(ARCH_FLAG) -mabi=ilp32
LINKER_SCRIPT_NAME ?= tensix.ld
LINKER_SCRIPT_SRC := $(TOOLCHAIN)/$(LINKER_SCRIPT_NAME)

OUTPUT_DIR ?= out
LINKER_SCRIPT := $(OUTPUT_DIR)/$(LINKER_SCRIPT_NAME)
FIRMWARE_NAME ?= firmware
INFO_NAME ?= $(FIRMWARE_NAME)
FIRMWARE_START ?= 0

# All objects are dumped into out, so we don't support two source files in different directories with the same name.
ifneq ($(words $(sort $(SOURCES))),$(words $(sort $(notdir $(SOURCES)))))
$(error $$(SOURCES) contains a duplicate filename)
endif

LDFLAGS := $(ARCH) $(OPT_FLAGS) -Wl,--gc-sections -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -Wl,--defsym=__firmware_start=$(FIRMWARE_START) \
-T$(LINKER_SCRIPT) -L$(TOOLCHAIN) -nostartfiles

OUTFW := $(OUTPUT_DIR)/$(FIRMWARE_NAME)
CKERNEL_DEPS := $(wildcard $(COMPILED_KERNELS_DIR)/*.d)


OBJECTS := $(addprefix $(CKERNELS_COMMON_OUT_DIR)/, $(addsuffix .o,$(basename $(notdir $(SOURCES)))))
# FIXME MT: Remove deps to allow build of common files into different output folder. Think about this again
# DEPENDS := $(addprefix $(CKERNELS_COMMON_OUT_DIR)/, $(addsuffix .d,$(basename $(notdir $(SOURCES)))))


# vpath % $(subst $(space),:,$(TOOLCHAIN) $(SOURCE_DIRS) $(TT_METAL_HOME)/src/ckernels/$(ARCH_NAME)/common/out)

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/wormhole"
else
  INCLUDES += -I "$(TT_METAL_HOME)/tt_metal/src/firmware/riscv/grayskull"
endif

#FIXME: removing bin generation for now, as it is not used
ifeq ($(MAKE_FW_MAP),1)
all:: extras $(OUTFW).hex  $(OUTFW).map #$(OUTFW).bin
	@$(PRINT_SUCCESS)
else
all:: extras $(OUTFW).hex #$(OUTFW).bin
	@$(PRINT_SUCCESS)
endif

$(OUTPUT_DIR):
	-mkdir -p $@

$(LINKER_SCRIPT): $(LINKER_SCRIPT_SRC) | $(OUTPUT_DIR)
	$(CXX) $(DEFINES) $(DEFS) $(INCLUDES) -E -P -x c -o $@ $<

# Link using C++ compiler
$(OUTFW).elf: $(OBJECTS) $(EXTRA_OBJECTS) $(LINKER_SCRIPT)
	# mkdir -p $(OUTPUT_DIR)
	@echo "$(CXX) $(LDFLAGS) -o $@ $(filter-out $(LINKER_SCRIPT),$^)"
	$(CXX) $(LDFLAGS) -o $@ $(filter-out $(LINKER_SCRIPT),$^)
	chmod -x $@

# Generate hex-formatted firmware for LUA
$(OUTFW).hex: $(OUTFW).elf
	$(OBJCOPY) -O verilog $< $@.tmp
	python3 $(TOOLCHAIN)/hex8tohex32.py $@.tmp > $@
	rm $@.tmp

$(OUTFW).bin: $(OUTFW).elf
	echo $(OBJCOPY) -O binary $< $@
	$(OBJCOPY) -O binary $< $@

# Symbol map
$(OUTFW).map: $(OUTFW).elf
	$(OBJDUMP) -dSxCltg $< > $@

# Create a file that maps where we log in firmware source to what is being logged
# IMPROVE: handle multiple source files
$(OUTPUT_DIR)/$(INFO_NAME).fwlog: $(OUTFW).elf
	python3 $(FIRMWARE_RISCV_DIR)/toolchain/fwlog.py --depfile $(OUTPUT_DIR)/$(INFO_NAME).d > $@

$(OUTPUT_DIR)/ckernel.fwlog: $(OUTFW).elf $(CKERNEL_DEPS) $(CKERNELS_COMMON_OUT_DIR)/ckernel.d
	python3 $(FIRMWARE_RISCV_DIR)/toolchain/fwlog.py --depfile $(OUTPUT_DIR)/ckernel.d > $@.tmp
#MT: FIXME	python3 $(FIRMWARE_RISCV_DIR)/toolchain/fwlog.py --depfile $(CKERNEL_DEPS) --path=$(CKERNELS)/src >> $@.tmp
#	common .cc files that are not in any dependenecies
	python3 $(FIRMWARE_RISCV_DIR)/toolchain/fwlog.py --depfile $(CKERNELS)/$(ARCH_NAME)/common/src/fwlog_list --path=$(CKERNELS_COMMON_DIR)/src >> $@.tmp
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
	$(RISCV_TOOLS_PREFIX)objdump -dSxCltg $< > $@

clean2::
	rm $(OUTFW).elf $(OUTFW).hex $(OUTFW).bin $(OUTFW).map $(SILENT_ERRORS)
	rm $(OBJECTS) $(SILENT_ERRORS)
	# rm $(DEPENDS) $(SILENT_ERRORS)
	rm $(EXTRA_OBJECTS) $(SILENT_ERRORS)
	rm $(EXTRA_OBJECTS:.o=.d) $(SILENT_ERRORS)
	rmdir $(OUTPUT_DIR) $(SILENT_ERRORS)
	rm $(OUTPUT_DIR)/$(INFO_NAME).fwlog $(OUTPUT_DIR)/$(INFO_NAME)-decodedline.txt $(OUTPUT_DIR)/$(INFO_NAME)-debuginfo.txt $(OUTPUT_DIR)/$(INFO_NAME)-symbols.txt $(SILENT_ERRORS)
	rm -rf $(GEN_DIR) $(SILENT_ERRORS)

extras::

.PHONY: clean2 all extras

# -include $(DEPENDS)
