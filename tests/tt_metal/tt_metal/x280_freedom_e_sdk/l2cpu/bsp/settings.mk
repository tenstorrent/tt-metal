# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# freedom-e-sdk BSP settings for a SiFive X280 hart in a Blackhole L2CPU tile.
#
# These values are not a guess: they are exactly what tt-llm-engine's
# x280/Makefile uses to build the bare-metal firmware that already runs on this
# hardware --
#
#   ARCH_FLAGS := -march=rv64gc -mabi=lp64d
#   CFLAGS     := $(ARCH_FLAGS) ... -mcmodel=medany
#
# and they are also a legal freedom-e-sdk BSP configuration, because the X280 is
# a stock SiFive core. That is the whole compatibility story in four lines.
#
# RISCV_SERIES matters: freedom-metal's src/entry.S clears the SiFive Feature
# Disable CSR 0x7c1 (the "chicken bit") only when the BSP's linker script
# PROVIDEs __metal_chicken_bit = 1, which the sifive-* series BSPs do. That is
# the same bring-up step tt-llm-engine's x280/boot/entry.S performs as its
# "Step 3", so the two boot paths agree.

RISCV_ARCH = rv64gc
RISCV_ABI = lp64d

# medany matches tt-llm-engine. LIM sits at 0x08001000, which is comfortably
# inside medlow's range too, so this is belt and braces rather than a
# requirement -- unlike the Quasar DM port next door, where the upstream rv64
# BSP's 0x80000000 base was out of medlow reach entirely.
RISCV_CMODEL = medany
RISCV_SERIES = sifive-7-series

TARGET_TAGS = tt-x280 l2cpu
TARGET_DHRY_ITERS = 20000000
TARGET_CORE_ITERS = 5000
TARGET_FREERTOS_WAIT_MS = 1000
TARGET_INTR_WAIT_CYCLE = 0

# A real X280 is rv64gcv with VLEN=512. The vendored toolchain accepts
# -march=rv64gcv, but its newlib is built non-multilib for rv64gc/lp64d, so the
# base ISA here stays rv64gc to match both the libc and tt-llm-engine's
# firmware. Vector code is opted into per translation unit; src/x280_bringup.c
# enables mstatus.VS so the unit is usable at all.
