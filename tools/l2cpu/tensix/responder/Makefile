# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
#
# Link test responder for the x280 harts of an L2CPU tile: build/responder.bin, position independent, loaded at
# channel_base + L2CPU_LINK_IMAGE_OFFSET (kernels/l2cpu_link.h). Needs riscv64-unknown-elf-gcc.
CROSS ?= riscv64-unknown-elf-
CFLAGS = -march=rv64ima_zicsr_zifencei -mabi=lp64 -mcmodel=medany -O2 -fno-jump-tables -fno-pic \
         -ffreestanding -nostdlib -fno-builtin -Wall -Wextra
B = build
all: $(B)/responder.bin
$(B)/responder.elf: start.S responder.c link.ld ../kernels/l2cpu_link.h ../../include/l2cpu_hw.h
	@mkdir -p $(B)
	$(CROSS)gcc $(CFLAGS) -T link.ld -o $@ start.S responder.c
$(B)/responder.bin: $(B)/responder.elf
	$(CROSS)objcopy -O binary $< $@
clean:
	rm -rf $(B)
.PHONY: all clean
