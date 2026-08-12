// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Runtime dest-accumulation-mode tracker for assert-only dest-capacity checks.
// Exists only when ENABLE_LLK_ASSERT is set; production builds never name the
// symbol. Call sites read via LLK_ASSERT_DEST_ACC_MODE() so the variable is not
// referenced outside assert builds. Writers update via LLK_ASSERT_SET_DEST_ACC_MODE
// at shared APIs (enable/disable_fp32_dest_acc, compute_kernel_hw_startup), outside
// UNPACK/MATH/PACK so every TRISC thread updates its own copy.
#ifdef ENABLE_LLK_ASSERT
// C++17 inline variable: emitted only in TUs that ODR-use it, weak/merged at
// link; per-thread storage since each TRISC thread is a separate build.
inline bool dst_fp32_acc_en = false;
#define LLK_ASSERT_DEST_ACC_MODE() (dst_fp32_acc_en)
#define LLK_ASSERT_SET_DEST_ACC_MODE(enable) \
    do                                       \
    {                                        \
        dst_fp32_acc_en = static_cast<bool>(enable); \
    } while (0)
#else
// Production LLK_ASSERT is ((void)sizeof(cond)) — the condition is unevaluated,
// so a constant placeholder compiles fine and the tracker symbol is never named.
#define LLK_ASSERT_DEST_ACC_MODE() (false)
#define LLK_ASSERT_SET_DEST_ACC_MODE(enable) ((void)0)
#endif

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
