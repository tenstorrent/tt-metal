// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// First in every TRISC unit of a profiler build. TRISC timing depends on the code address, so run_kernel gets a 2 KiB
// aligned section that sections.ld places after the harness code, and harness changes cannot move it.
#if !defined(ARCH_QUASAR)
// The type RUNTIME_PARAMETERS expands to; build.h is not included because some kernels define RuntimeParams.
struct RuntimeParams;

__attribute__((noinline, section(".text.run_kernel"), aligned(2048))) void run_kernel(const struct RuntimeParams& params);
#if defined(LLK_DBG_BARRIER) && !defined(LLK_PERF_OOL)
// The callees of run_kernel start on a 1 KiB boundary (sections.ld reads llk_text_tail_align), so a size change of
// run_kernel does not move them, then llk_loop_end_pad bytes of never executed NOPs, an assembler symbol the Wormhole
// perf layout sets at link time like llk_loop_pad (barrier.h).
asm(".globl llk_text_tail_align\n"
    ".set llk_text_tail_align, 1024\n"
    ".pushsection .text.run_kernel.tail,\"ax\",@progbits\n"
    ".ifndef llk_loop_end_pad\n"
    ".set llk_loop_end_pad, 0\n"
    ".endif\n"
    ".rept llk_loop_end_pad / 4\n"
    "nop\n"
    ".endr\n"
    ".popsection");
#endif
#endif
