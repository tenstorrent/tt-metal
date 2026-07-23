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

// L1 access-range asserts (debug-only; wrap LLK_ASSERT, so elided in release builds).
//
// Validate that BOTH the base and the end of a multi-word / strided L1 access lie inside valid L1,
// extending the base-only is_valid_L1_address() check. All addresses are LLK-transformed 16B-word
// addresses (the same space is_valid_L1_address() consumes); the range length is likewise in 16B
// words, NOT raw bytes. is_valid_L1_address() must be in scope at the call site (include
// llk_memory_checks.h).

/**
 * @brief Assert that an L1 access range [base, base + size) lies fully within valid L1.
 * @param base_address First accessed 16B-word address.
 * @param size Range length in 16B words (assumed >= 1).
 * @param message Diagnostic message.
 * @note Checks both the base and the last word (base + size - 1).
 */
#define LLK_ASSERT_L1_RANGE(base_address, size, message) \
    LLK_ASSERT(is_valid_L1_address(base_address) && is_valid_L1_address((base_address) + (size) - 1), message)

/**
 * @brief Assert an L1 access whose last accessed word is not simply base + size (strided /
 *        multi-face); the caller supplies the last accessed word explicitly.
 * @param base_address First accessed 16B-word address.
 * @param last_address Last accessed 16B-word address.
 * @param message Diagnostic message.
 */
#define LLK_ASSERT_L1_RANGE_BASE_LAST(base_address, last_address, message) \
    LLK_ASSERT(is_valid_L1_address(base_address) && is_valid_L1_address(last_address), message)
