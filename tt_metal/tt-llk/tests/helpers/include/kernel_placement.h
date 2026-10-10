// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// First in every TRISC unit of a profiler build. TRISC timing depends on the code address, so run_kernel gets a 2 KiB
// aligned section that sections.ld places after the harness code, and harness changes cannot move it.
#if !defined(ARCH_QUASAR)
// The type RUNTIME_PARAMETERS expands to; build.h is not included because some kernels define RuntimeParams.
struct RuntimeParams;

#if defined(LLK_PERF_INIT_ONLY) // INIT measurement build: INIT runs from fixed sections ahead of main, run_kernel's place is free
__attribute__((noinline, section(".text.run_kernel"))) void run_kernel(const struct RuntimeParams& params);
#else
__attribute__((noinline, section(".text.run_kernel"), aligned(2048))) void run_kernel(const struct RuntimeParams& params);
#endif
#if defined(LLK_DBG_BARRIER) && !defined(LLK_PERF_INIT_ONLY)
// run_kernel's callees start on the thread's period (sections.ld), so run_kernel's size cannot move them, after
// llk_loop_end_pad bytes of NOPs that perf/layout.py sets at link time (the OOL threads also pad in profiler.h).
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
#define LLK_TEXT_TAIL_ALIGN_ "512"
#else
#define LLK_TEXT_TAIL_ALIGN_ "1024"
#endif
asm(".globl llk_text_tail_align\n"
    ".set llk_text_tail_align, " LLK_TEXT_TAIL_ALIGN_
    "\n"
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
