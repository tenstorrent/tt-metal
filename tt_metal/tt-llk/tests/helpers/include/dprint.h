// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// To use device print in an LLK test, run pytest with --logging-level=debug
// (or trace); that flips -DDEBUG_PRINT_ENABLED on for every variant. All print
// calls compile to nothing when the macro is not set.

// PROCESSOR_INDEX, LLK_DEVICE_PRINT_BUFFER_BASE, LLK_RUNTIME_ARGS_START,
// DEVICE_PRINT_BUFFER_SIZE and DEVICE_PRINT_BUFFER_SIZE2 (the Quasar DM buffer)
// are passed in by test_config.py at build time.

// Disabled under COVERAGE: coverage linker scripts grow TRISC sections
// way past the device print buffer slot, so they can't share L1.
// The alternative would require a lot more hacks; the only proper solution
// is fixing the LLK infra memory layout across the board.

#pragma once

#if defined(DEBUG_PRINT_ENABLED) && !defined(COVERAGE)

#include <cstdint>

#include "dev_mem_map.h"

// Device print occasionally writes from host; Blackhole needs a fence,
// and Quasar on DM cores needs a cache flush on top of that.
// Informed by tt_metal/hw/inc/internal/tt-{1,2}xx/risc_common.h.
// The original headers are too heavy to be included from LLK infra.
inline __attribute__((always_inline)) void invalidate_l1_cache()
{
#ifdef ARCH_BLACKHOLE
    __asm__ __volatile__("fence" ::: "memory");
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    __asm__ __volatile__("tt.cache.cflush.d.l1 x0" ::: "memory");
    __asm__ __volatile__("fence" ::: "memory");
#endif
}

// We need to include this header after the above definitions.
#include "api/debug/device_print.h"

#if !defined(LLK_DEVICE_PRINT_BUFFER_BASE) || !defined(LLK_RUNTIME_ARGS_START)
#error "LLK_DEVICE_PRINT_BUFFER_BASE and LLK_RUNTIME_ARGS_START must be defined by the build"
#endif

// On Quasar the LLK buffer base is handed to the build in the uncached L1 alias region
// (the upper 4 MB at MEM_L1_UNCACHED_BASE; see kernel_buffer_base in test_config.py),
// so its numeric value carries that offset. Strip it to get the physical L1 address
// for the RUNTIME_ARGS overlap check below.
#if defined(ARCH_QUASAR)
constexpr uintptr_t llk_device_print_buffer_l1_base = LLK_DEVICE_PRINT_BUFFER_BASE - MEM_L1_UNCACHED_BASE;
#else
constexpr uintptr_t llk_device_print_buffer_l1_base = LLK_DEVICE_PRINT_BUFFER_BASE;
#endif
// Check the footprint of the device print region against RUNTIME_ARGS, the next region.
static_assert(
    llk_device_print_buffer_l1_base + sizeof(DevicePrintMemoryLayout) <= LLK_RUNTIME_ARGS_START,
    "LLK device print buffer overlaps RUNTIME_ARGS; adjust TestConfig.DEVICE_PRINT_BUFFER_BASE/"
    "DEVICE_PRINT_BUFFER_SIZE/DEVICE_PRINT_BUFFER_SIZE2 in tests/python_tests/helpers/test_config.py.");

// A single #include "dprint.h" exposes every device print facility.
#include "api/debug/dprint_tile.h"
#include "dprint_tensix.h"
#else

#define DEVICE_PRINT(fmt, ...)

#endif // defined(DEBUG_PRINT_ENABLED) && !defined(COVERAGE)
