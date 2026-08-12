# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Quasar DM BSP: ISA from `riscv-tt-elf-gcc -mcpu=tt-qsr64-rocc` (no F/D/C/V).
# xttcache = CFLUSH.D.L1 family; xttroccqsr = Quasar RoCC.

RISCV_ARCH = rv64imab_zihpm_zmmul_zaamo_zalrsc_zba_zbb_zbc_zbs_xttcache_xttroccqsr
RISCV_ABI = lp64
RISCV_CMODEL = medany
RISCV_SERIES = None

# -mcpu selects qsr64-lp64 multilib; -march alone picks the wrong libc.
RISCV_CFLAGS += -mcpu=tt-qsr64-rocc
RISCV_CXXFLAGS += -mcpu=tt-qsr64-rocc
RISCV_ASFLAGS += -mcpu=tt-qsr64-rocc
RISCV_CCASFLAGS += -mcpu=tt-qsr64-rocc

TARGET_TAGS = tt-quasar
TARGET_DHRY_ITERS = 20000000
TARGET_CORE_ITERS = 5000
TARGET_FREERTOS_WAIT_MS = 1000
TARGET_INTR_WAIT_CYCLE = 0
