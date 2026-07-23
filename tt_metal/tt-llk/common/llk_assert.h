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

#define LLK_ASSERT(condition, message) ASSERT(condition)

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

// Validates a half-open L1 range [base, last] against is_valid_L1_address, both endpoints in the
// LLK-transformed 16B-word address format is_valid_L1_address expects. Use when an LLK API reads/writes
// beyond its base address (multi-face unpackers, strided packers, MOP-driven bursts) and the last accessed
// address is a compile-visible expression that is not simply base + size (e.g. a stride times an addend
// count). Callers must `#include "llk_memory_checks.h"` for is_valid_L1_address.
#define LLK_ASSERT_L1_RANGE_BASE_LAST(base, last, message) \
    do                                                     \
    {                                                      \
        LLK_ASSERT(is_valid_L1_address(base), message);    \
        LLK_ASSERT(is_valid_L1_address(last), message);    \
    } while (0)

// Validates a half-open L1 range [base, base + size_16B_words - 1], both in LLK-transformed 16B-word units.
#define LLK_ASSERT_L1_RANGE(base, size_16B_words, message) LLK_ASSERT_L1_RANGE_BASE_LAST(base, (base) + (size_16B_words) - 1, message)
