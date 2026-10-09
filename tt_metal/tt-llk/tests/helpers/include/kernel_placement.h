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
// The callees of run_kernel start on a boundary of the thread's period (sections.ld reads llk_text_tail_align: 1 KiB,
// math 512 B), so a size change of run_kernel does not move them (also on the out of line threads: the linker places
// the code after run_kernel by its size before relaxation, so code before the TILE_LOOP park moved the callees), then
// llk_loop_end_pad bytes of never executed NOPs, an assembler symbol the Wormhole perf layout sets at link time like
// llk_loop_pad (barrier.h). The out of line threads also put llk_loop_end_pad after the TILE_LOOP end read
// (profiler.h): there it opens a gap between the loop and the blocks the compiler places after the zone end, which the
// matmul math needs to stay clear of its icache conflicts, while the copy here keeps the SFPU callees tunable.
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
#define LLK_TEXT_TAIL_ALIGN_ "512"
#else
#define LLK_TEXT_TAIL_ALIGN_ "1024"
#endif
asm(".globl llk_text_tail_align\n"
    ".set llk_text_tail_align, " LLK_TEXT_TAIL_ALIGN_ "\n"
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
