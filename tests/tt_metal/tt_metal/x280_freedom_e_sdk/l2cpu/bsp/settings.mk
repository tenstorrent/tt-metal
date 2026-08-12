# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# BSP settings matching tt-llm-engine x280/Makefile (-march=rv64gc -mabi=lp64d
# -mcmodel=medany). sifive-* series PROVIDEs __metal_chicken_bit=1 so entry.S
# clears Feature Disable CSR 0x7c1 (same as tt-llm-engine boot Step 3).

RISCV_ARCH = rv64gc
RISCV_ABI = lp64d

# Match tt-llm-engine; LIM at 0x08001000 is in medlow range too.
RISCV_CMODEL = medany
RISCV_SERIES = sifive-7-series

TARGET_TAGS = tt-x280 l2cpu
TARGET_DHRY_ITERS = 20000000
TARGET_CORE_ITERS = 5000
TARGET_FREERTOS_WAIT_MS = 1000
TARGET_INTR_WAIT_CYCLE = 0

# Silicon is rv64gcv; newlib is non-multilib rv64gc/lp64d so keep base ISA.
# Opt into vector per TU; x280_bringup.c enables mstatus.VS.
