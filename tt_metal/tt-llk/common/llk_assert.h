// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_LLK_ASSERT

#define LLK_ASSERT_BLOCK(block_call) \
    do                               \
    {                                \
        (void)(block_call);          \
    } while (0)

#if defined(ENV_LLK_INFRA) || defined(ENABLE_LLK_ASSERT_ONLY)

#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

#define LLK_ASSERT(condition, message) \
    do                                 \
    {                                  \
        if (UNLIKELY(!(condition)))    \
        {                              \
            asm volatile("ebreak");    \
        }                              \
    } while (0)

#else

// Assume we are executing in tt-metal and we have assert already available.
#include "api/debug/assert.h"

// DEBUG (assert locator): the watcher reports a raw __LINE__ but cannot resolve the file, so a failing
// LLK_ASSERT shows a misleading "line N / <current kernel>" pair. DPRINT the assert's OWN message + line on
// failure so the exact assert is identifiable on the emulator (enable with TT_METAL_DPRINT_CORES). Guarded
// by KERNEL_BUILD so host / non-kernel TUs keep the plain ASSERT. `message` is always a string literal, so it
// concatenates into the DPRINT format string. Remove after.
#if defined(KERNEL_BUILD)
#include "api/debug/dprint.h"
#define LLK_ASSERT(condition, message)                                         \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            DPRINT("LLK_ASSERT_FAIL: " message " @L{}\n", (unsigned)__LINE__); \
        }                                                                      \
        ASSERT(condition);                                                     \
    } while (0)
#else
#define LLK_ASSERT(condition, message) ASSERT(condition)
#endif

#endif // defined(ENV_LLK_INFRA) || defined(ENABLE_LLK_ASSERT_ONLY)

#else

// sizeof creates an unevaluated context: the condition is fully compiled
// (type-checked, name-resolved) but never executed at runtime.
#define LLK_ASSERT(condition, message) ((void)sizeof((condition)))

// Comma with 0: block_call is type-checked (incl. void returns); void is
// not a valid sizeof operand, so (void) cast then discard via comma.
#define LLK_ASSERT_BLOCK(block_call)                     \
    do                                                   \
    {                                                    \
        (void)sizeof(decltype(((void)(block_call), 0))); \
    } while (0)

#endif // ENABLE_LLK_ASSERT

// Inverse of LLK_ASSERT: Triggers when the condition is true (failure condition)
#define LLK_PANIC(condition, message) LLK_ASSERT(!(condition), message)
