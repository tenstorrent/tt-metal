# Makefile - X280 bare-metal step-1 heartbeat FW.
#
# Builds boot/entry.S + src/counter.c into build/counter.bin (raw binary the
# host NOC-writes into LIM at 0x08000000). Auto-fetches a matched
# riscv64-unknown-elf GCC toolchain if not present.
#
# The toolchain is downloaded OUTSIDE the (mutagen-synced) tt-metal tree so
# its ~200 MB does not sync back to the Mac clone. Override TOOLCHAIN_DIR to
# point at an existing toolchain.

TOOLCHAIN_DIR ?= /localdev/$(USER)/x280-toolchain
TOOLCHAIN_URL := https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.04.05/riscv64-elf-ubuntu-22.04-gcc.tar.xz
CROSS         := $(TOOLCHAIN_DIR)/bin/riscv64-unknown-elf-

CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
NM      := $(CROSS)nm

ARCH_FLAGS := -march=rv64gc -mabi=lp64d
BARE_FLAGS := -nostdlib -nostartfiles -ffreestanding -mcmodel=medany
CFLAGS     := $(ARCH_FLAGS) $(BARE_FLAGS) -O2 -g -Wall

BUILD := build
LD_SCRIPT := ld/x280-lim.ld

.PHONY: all clean disasm symbols toolchain
all: $(BUILD)/counter.bin

$(BUILD)/entry.o: boot/entry.S | $(BUILD) toolchain
	$(CC) $(ARCH_FLAGS) -nostdlib -c -o $@ $<

$(BUILD)/counter.o: src/counter.c | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/counter.elf: $(BUILD)/entry.o $(BUILD)/counter.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/counter.o

$(BUILD)/counter.bin: $(BUILD)/counter.elf
	$(OBJCOPY) -O binary $< $@
	@echo "counter.bin: $$(wc -c < $@) bytes"

$(BUILD):
	mkdir -p $@

toolchain: $(CROSS)gcc

$(CROSS)gcc:
	@echo "Downloading RISC-V toolchain to $(TOOLCHAIN_DIR) ..."
	mkdir -p $(TOOLCHAIN_DIR)
	wget -q -O /tmp/x280-rv.tar.xz $(TOOLCHAIN_URL)
	tar -xf /tmp/x280-rv.tar.xz -C $(TOOLCHAIN_DIR) --strip-components=1
	rm -f /tmp/x280-rv.tar.xz
	@echo "Toolchain installed at $(TOOLCHAIN_DIR)"

disasm: $(BUILD)/counter.elf
	$(OBJDUMP) -d -S $<

symbols: $(BUILD)/counter.elf
	$(NM) $< | sort

clean:
	rm -rf $(BUILD)
