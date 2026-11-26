// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel code size: every phase a core runs shares the core's kernel config buffer (69 KB on BH Tensix for all
// five RISC binaries + runtime args + CB configs). With MK_OPT_SIZE defined, this file — included FIRST in a kernel
// source, before any API header, so the header-defined (always-inline) API code is covered too — compiles the kernel
// for size (~2.7x smaller TRISC binaries). no-gcse / no-schedule-insns{,2}: plain "Os" moved a %hi relocation away
// from its %lo pair, which the loader's XIP pass rejects (tt_elffile.cpp "R_RISCV_HI20 ... no matching R_RISCV_LO12").
// Arithmetic is unchanged (same SFPU / FPU instructions); speed is measured per phase (dev_kda_rec opt_size rows).
#pragma once

#if defined(MK_OPT_SIZE)
#pragma GCC optimize("Os", "no-gcse", "no-schedule-insns", "no-schedule-insns2")
#endif
