// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// To use device print in an LLK test, build with -DDEBUG_PRINT_ENABLED
// by setting device_print_build=DevicePrintBuild.Yes in TestConfig.
// All print calls compile to nothing when the macro is not set.

#pragma once

#ifdef DEBUG_PRINT_ENABLED

#include <cstdint>

#if defined(LLK_TRISC_UNPACK)
#define PROCESSOR_INDEX 2
#elif defined(LLK_TRISC_MATH)
#define PROCESSOR_INDEX 3
#elif defined(LLK_TRISC_PACK)
#define PROCESSOR_INDEX 4
#elif defined(LLK_TRISC_ISOLATE_SFPU)
#define PROCESSOR_INDEX 5 // Quasar
#endif

// Mirrored in tests/python_tests/helpers/test_config.py.
#define LLK_DEVICE_PRINT_BUFFER_BASE 0x13000 // Consumed by dprint_buffer.h
#define DPRINT_BUFFER_SIZE           1024    // Overrides dprint_common.h

#define USE_DEVICE_PRINT

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

// This header has to be included after the above definitions.
#include "api/debug/device_print.h"

#else

#define DEVICE_PRINT(fmt, ...)
#define DEVICE_PRINT_INITIALIZE_LOCK()

#endif // DEBUG_PRINT_ENABLED
