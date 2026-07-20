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
# -fno-tree-loop-distribute-patterns: stop GCC turning init loops into
# memset/memcpy calls (we have no libc under -nostdlib).
CFLAGS     := $(ARCH_FLAGS) $(BARE_FLAGS) -O2 -g -Wall -Iinclude -fno-tree-loop-distribute-patterns

BUILD := build
LD_SCRIPT := ld/x280-lim.ld

.PHONY: all clean disasm symbols toolchain
all: $(BUILD)/counter.bin $(BUILD)/poller.bin $(BUILD)/dma_probe.bin $(BUILD)/grid_drain.bin $(BUILD)/grid_drain4.bin $(BUILD)/poll4.bin $(BUILD)/noc1_probe.bin $(BUILD)/poll4n1.bin $(BUILD)/poll6n1.bin $(BUILD)/pollmp.bin $(BUILD)/gridilp.bin $(BUILD)/d2hbw.bin $(BUILD)/profrelay.bin $(BUILD)/profcons.bin $(BUILD)/profcons_split.bin $(BUILD)/eccscrub.bin $(BUILD)/rdrbench.bin

$(BUILD)/rdrbench.o: src/rdrbench.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/rdrbench.elf: $(BUILD)/entry.o $(BUILD)/rdrbench.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/rdrbench.o

$(BUILD)/rdrbench.bin: $(BUILD)/rdrbench.elf
	$(OBJCOPY) -O binary $< $@
	@echo "rdrbench.bin: $$(wc -c < $@) bytes"

# profll: per-lane lossless drainer (step-0 active FW; handoff skeleton + profcons per-lane drain).
$(BUILD)/profll.o: src/profll.c include/noc.h include/x280_boot.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profll.elf: $(BUILD)/entry.o $(BUILD)/profll.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profll.o

$(BUILD)/profll.bin: $(BUILD)/profll.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profll.bin: $$(wc -c < $@) bytes"

# profstream: linearized pipeline (2 readers + 1 relay + single host ring), sticky-src identity.
$(BUILD)/profstream.o: src/profstream.c include/noc.h include/x280_boot.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profstream.elf: $(BUILD)/entry.o $(BUILD)/profstream.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profstream.o

$(BUILD)/profstream.bin: $(BUILD)/profstream.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profstream.bin: $$(wc -c < $@) bytes"

# lim_idle: resident idle / boot-handoff FW at 0x08000000 (its own LD script; ORIGIN differs).
$(BUILD)/lim_idle.o: src/lim_idle.c include/noc.h include/x280_boot.h | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/lim_idle.elf: $(BUILD)/entry.o $(BUILD)/lim_idle.o ld/x280-lim-idle.ld | toolchain
	$(CC) -nostdlib -T ld/x280-lim-idle.ld $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/lim_idle.o

$(BUILD)/lim_idle.bin: $(BUILD)/lim_idle.elf
	$(OBJCOPY) -O binary $< $@
	@echo "lim_idle.bin: $$(wc -c < $@) bytes"

$(BUILD)/eccscrub.o: src/eccscrub.c | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/eccscrub.elf: $(BUILD)/entry.o $(BUILD)/eccscrub.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/eccscrub.o

$(BUILD)/eccscrub.bin: $(BUILD)/eccscrub.elf
	$(OBJCOPY) -O binary $< $@
	@echo "eccscrub.bin: $$(wc -c < $@) bytes"

$(BUILD)/eccpoke.o: src/eccpoke.c | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/eccpoke.elf: $(BUILD)/entry.o $(BUILD)/eccpoke.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/eccpoke.o

$(BUILD)/eccpoke.bin: $(BUILD)/eccpoke.elf
	$(OBJCOPY) -O binary $< $@
	@echo "eccpoke.bin: $$(wc -c < $@) bytes"

$(BUILD)/entry.o: boot/entry.S | $(BUILD) toolchain
	$(CC) $(ARCH_FLAGS) -nostdlib -c -o $@ $<

$(BUILD)/counter.o: src/counter.c | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/counter.elf: $(BUILD)/entry.o $(BUILD)/counter.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/counter.o

$(BUILD)/counter.bin: $(BUILD)/counter.elf
	$(OBJCOPY) -O binary $< $@
	@echo "counter.bin: $$(wc -c < $@) bytes"

$(BUILD)/poller.o: src/poller.c include/noc.h | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/poller.elf: $(BUILD)/entry.o $(BUILD)/poller.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/poller.o

$(BUILD)/poller.bin: $(BUILD)/poller.elf
	$(OBJCOPY) -O binary $< $@
	@echo "poller.bin: $$(wc -c < $@) bytes"

$(BUILD)/dma_probe.o: src/dma_probe.c include/dma_engine.h include/noc.h | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/dma_probe.elf: $(BUILD)/entry.o $(BUILD)/dma_probe.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/dma_probe.o

$(BUILD)/dma_probe.bin: $(BUILD)/dma_probe.elf
	$(OBJCOPY) -O binary $< $@
	@echo "dma_probe.bin: $$(wc -c < $@) bytes"

$(BUILD)/grid_drain.o: src/grid_drain.c include/dma_engine.h include/noc.h | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/grid_drain.elf: $(BUILD)/entry.o $(BUILD)/grid_drain.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/grid_drain.o

$(BUILD)/grid_drain.bin: $(BUILD)/grid_drain.elf
	$(OBJCOPY) -O binary $< $@
	@echo "grid_drain.bin: $$(wc -c < $@) bytes"

$(BUILD)/grid_drain4.o: src/grid_drain4.c include/dma_engine.h include/noc.h | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/grid_drain4.elf: $(BUILD)/entry.o $(BUILD)/grid_drain4.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/grid_drain4.o

$(BUILD)/grid_drain4.bin: $(BUILD)/grid_drain4.elf
	$(OBJCOPY) -O binary $< $@
	@echo "grid_drain4.bin: $$(wc -c < $@) bytes"

# poll4 uses RVV (vle64.v) to read a flit in one transaction -> vector arch.
VEC_CFLAGS := -march=rv64gcv_zfh_zba_zbb -mabi=lp64d $(BARE_FLAGS) -O2 -g -Wall -Iinclude -fno-tree-loop-distribute-patterns
$(BUILD)/poll4.o: src/poll4.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/poll4.elf: $(BUILD)/entry.o $(BUILD)/poll4.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/poll4.o

$(BUILD)/poll4.bin: $(BUILD)/poll4.elf
	$(OBJCOPY) -O binary $< $@
	@echo "poll4.bin: $$(wc -c < $@) bytes"

$(BUILD)/noc1_probe.o: src/noc1_probe.c include/dma_engine.h include/noc.h | $(BUILD) toolchain
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/noc1_probe.elf: $(BUILD)/entry.o $(BUILD)/noc1_probe.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/noc1_probe.o

$(BUILD)/noc1_probe.bin: $(BUILD)/noc1_probe.elf
	$(OBJCOPY) -O binary $< $@
	@echo "noc1_probe.bin: $$(wc -c < $@) bytes"

$(BUILD)/poll4n1.o: src/poll4n1.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/poll4n1.elf: $(BUILD)/entry.o $(BUILD)/poll4n1.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/poll4n1.o

$(BUILD)/poll4n1.bin: $(BUILD)/poll4n1.elf
	$(OBJCOPY) -O binary $< $@
	@echo "poll4n1.bin: $$(wc -c < $@) bytes"

$(BUILD)/poll6n1.o: src/poll6n1.c include/dma_engine.h include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/poll6n1.elf: $(BUILD)/entry.o $(BUILD)/poll6n1.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/poll6n1.o

$(BUILD)/poll6n1.bin: $(BUILD)/poll6n1.elf
	$(OBJCOPY) -O binary $< $@
	@echo "poll6n1.bin: $$(wc -c < $@) bytes"

$(BUILD)/pollmp.o: src/pollmp.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/pollmp.elf: $(BUILD)/entry.o $(BUILD)/pollmp.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/pollmp.o

$(BUILD)/pollmp.bin: $(BUILD)/pollmp.elf
	$(OBJCOPY) -O binary $< $@
	@echo "pollmp.bin: $$(wc -c < $@) bytes"

$(BUILD)/gridilp.o: src/gridilp.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/gridilp.elf: $(BUILD)/entry.o $(BUILD)/gridilp.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/gridilp.o

$(BUILD)/gridilp.bin: $(BUILD)/gridilp.elf
	$(OBJCOPY) -O binary $< $@
	@echo "gridilp.bin: $$(wc -c < $@) bytes"

$(BUILD)/d2hbw.o: src/d2hbw.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/d2hbw.elf: $(BUILD)/entry.o $(BUILD)/d2hbw.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/d2hbw.o

$(BUILD)/d2hbw.bin: $(BUILD)/d2hbw.elf
	$(OBJCOPY) -O binary $< $@
	@echo "d2hbw.bin: $$(wc -c < $@) bytes"

$(BUILD)/profrelay.o: src/profrelay.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profrelay.elf: $(BUILD)/entry.o $(BUILD)/profrelay.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profrelay.o

$(BUILD)/profrelay.bin: $(BUILD)/profrelay.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profrelay.bin: $$(wc -c < $@) bytes"

$(BUILD)/profcons.o: src/profcons.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profcons.elf: $(BUILD)/entry.o $(BUILD)/profcons.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profcons.o

$(BUILD)/profcons.bin: $(BUILD)/profcons.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profcons.bin: $$(wc -c < $@) bytes"

$(BUILD)/profcons_split.o: src/profcons_split.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profcons_split.elf: $(BUILD)/entry.o $(BUILD)/profcons_split.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profcons_split.o

$(BUILD)/profcons_split.bin: $(BUILD)/profcons_split.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profcons_split.bin: $$(wc -c < $@) bytes"

$(BUILD)/profsock.o: src/profsock.c include/noc.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profsock.elf: $(BUILD)/entry.o $(BUILD)/profsock.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profsock.o

$(BUILD)/profsock.bin: $(BUILD)/profsock.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profsock.bin: $$(wc -c < $@) bytes"

# profzone: PRODUCTION drainer -- an exact copy of profstream's linearized pipeline, kept separate so
# profstream stays frozen as the benchmark reference while profzone takes real-marker/production changes.
$(BUILD)/profzone.o: src/profzone.c include/noc.h include/x280_boot.h | $(BUILD) toolchain
	$(CC) $(VEC_CFLAGS) -c -o $@ $<

$(BUILD)/profzone.elf: $(BUILD)/entry.o $(BUILD)/profzone.o $(LD_SCRIPT) | toolchain
	$(CC) -nostdlib -T $(LD_SCRIPT) $(ARCH_FLAGS) -o $@ $(BUILD)/entry.o $(BUILD)/profzone.o

$(BUILD)/profzone.bin: $(BUILD)/profzone.elf
	$(OBJCOPY) -O binary $< $@
	@echo "profzone.bin: $$(wc -c < $@) bytes"

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
