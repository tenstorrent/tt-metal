# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# freedom-e-sdk BSP settings for a Quasar DM core.
#
# freedom-e-sdk's scripts/standalone.mk turns RISCV_ARCH / RISCV_ABI /
# RISCV_CMODEL into -march / -mabi / -mcmodel, and appends whatever a BSP adds
# to RISCV_*FLAGS. That is the whole hook needed to retarget it at Tenstorrent's
# toolchain.
#
# The -march string is what `riscv-tt-elf-gcc -mcpu=tt-qsr64-rocc -Q
# --help=target` reports, i.e. exactly the ISA tt-metal builds Quasar DM
# firmware for. Note what is NOT in it: no F/D (soft-float only) and no C
# (no compressed instructions). A stock SiFive X280 is RV64GCV, so freedom-metal
# code that assumes hardware float or the V extension will not apply here.
#
# xttcache      -- the CFLUSH.D.L1 / CDISCARD.D.L1 family (X280 heritage)
# xttroccqsr    -- Quasar's RoCC accelerator interface

RISCV_ARCH = rv64imab_zihpm_zmmul_zaamo_zalrsc_zba_zbb_zbc_zbs_xttcache_xttroccqsr
RISCV_ABI = lp64

# medlow would be fine given the link addresses are all below 2 GiB, but medany
# keeps the BSP relocatable if the link base moves.
RISCV_CMODEL = medany
RISCV_SERIES = None

# -march alone selects the toolchain's default multilib, which is built for a
# different ISA. -mcpu=tt-qsr64-rocc selects the qsr64-lp64 multilib, so newlib
# and libgcc match. Both are passed; -mcpu drives multilib selection, -march
# pins the ISA.
RISCV_CFLAGS += -mcpu=tt-qsr64-rocc
RISCV_CXXFLAGS += -mcpu=tt-qsr64-rocc
RISCV_ASFLAGS += -mcpu=tt-qsr64-rocc
RISCV_CCASFLAGS += -mcpu=tt-qsr64-rocc

TARGET_TAGS = tt-quasar
TARGET_DHRY_ITERS = 20000000
TARGET_CORE_ITERS = 5000
TARGET_FREERTOS_WAIT_MS = 1000
TARGET_INTR_WAIT_CYCLE = 0
