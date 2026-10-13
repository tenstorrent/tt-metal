// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// First in every TRISC unit of a profiler build. TRISC timing depends on the code address, so run_kernel gets a 2 KiB
// aligned section that sections.ld places after the harness code, and harness changes cannot move it (Wormhole perf
// builds: no harness code ahead of it but the zone helpers).
#if !defined(ARCH_QUASAR)
// The type RUNTIME_PARAMETERS expands to; build.h is not included because some kernels define RuntimeParams.
struct RuntimeParams;

#if defined(LLK_PERF_INIT_ONLY) // INIT measurement build: INIT runs from fixed sections ahead of main, run_kernel's place is free
__attribute__((noinline, section(".text.run_kernel"))) void run_kernel(const struct RuntimeParams& params);
// INIT's sections start at this offset into the code, ahead of the harness (sections.ld), so harness growth cannot move
// them; the code starts with a jump to _start, where the loader starts the core
asm(".globl llk_init_region_at\n"
    ".set llk_init_region_at, 512\n"
    ".pushsection .text.llk_entry, \"ax\", @progbits\n"
    "j _start\n"
    ".popsection");
#elif defined(LLK_DBG_BARRIER) // Wormhole TILE_LOOP build
// run_kernel on its thread's period (the icache and predictor period, perf/layout.py) right after the zone helpers;
// the code no window runs (the harness, the kernel's other functions, INIT's out of line blocks) goes to an L1 area of
// its own (sections.ld .llk_spill, 12 KiB per TRISC above the code regions; the device print buffer's place when on)
__attribute__((noinline, section(".text.run_kernel"))) void run_kernel(const struct RuntimeParams& params);
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1
#define LLK_KERNEL_ALIGN_ "512"
#else
#define LLK_KERNEL_ALIGN_ "1024"
#endif
#if !defined(DEBUG_PRINT_ENABLED) && defined(COMPILE_FOR_TRISC)
#define LLK_SPILL_STR2_(x) #x
#define LLK_SPILL_STR_(x)  LLK_SPILL_STR2_(x)
asm(".globl llk_spill_at\n"
    ".set llk_spill_at, 0x12800 + 0x3000 * " LLK_SPILL_STR_(COMPILE_FOR_TRISC) "\n"
    ".globl llk_spill_size\n"
    ".set llk_spill_size, 0x3000");
#endif
asm(".globl llk_kernel_align\n"
    ".set llk_kernel_align, " LLK_KERNEL_ALIGN_
    "\n"
    ".pushsection .text.llk_entry, \"ax\", @progbits\n"
    "j _start\n"
    ".popsection");
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
