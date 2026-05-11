// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// If one wishes to use device print in an LLK test: include this header,
// and call DEVICE_PRINT_INITIALIZE_LOCK() before any DEVICE_PRINT calls.

#pragma once

// Informed by tt_metal/hw/inc/internal/tt-{1,2}xx/risc_common.h.
// The original headers are too heavy (and this is marginally more efficient.)
inline __attribute__((always_inline)) void invalidate_l1_cache()
{
#ifdef ARCH_BLACKHOLE
    __asm__ __volatile__("fence" ::: "memory");
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    __asm__ __volatile__("tt.cache.cflush.d.l1 x0" ::: "memory");
    __asm__ __volatile__("fence" ::: "memory");
#endif
}

// This mirrors hostdevcommon/api/hostdevcommon/dprint_common.h.
static constexpr uint32_t DPRINT_BUFFER_SIZE                = 204;
static constexpr uint32_t DEBUG_PRINT_SERVER_STARTING_MAGIC = 0x98989898;
static constexpr uint32_t DEBUG_PRINT_SERVER_DISABLED_MAGIC = 0xF8F8F8F8;
static constexpr uint32_t DEVICE_PRINT_RESET_BUFFER_MAGIC   = 0xF0E1D2C3;
static constexpr uint32_t DEVICE_PRINT_WRITE_STALL_FLAG     = 1u << 31;

#define DEBUG_PRINT_ENABLED
#define USE_DEVICE_PRINT

#ifdef LLK_TRISC_UNPACK
#define PROCESSOR_INDEX 2
#elif defined(LLK_TRISC_MATH)
#define PROCESSOR_INDEX 3
#elif defined(LLK_TRISC_PACK)
#define PROCESSOR_INDEX 4
#elif defined(LLK_TRISC_ISOLATE_SFPU)
#define PROCESSOR_INDEX 5 // Quasar
#endif

#define LLK_DEVICE_PRINT_BUFFER_BASE 0x16D400

#include "api/debug/device_print.h"
