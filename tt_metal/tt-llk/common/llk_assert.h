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

// Dest accumulation mode as last configured on this TRISC (math / pack hw_configure, _llk_set_fp32_dest_acc_).
// Kept in software rather than read from ALU_ACC_CTRL_Fp32_enabled: that bit is MATH-owned, toggled transiently
// (transpose_dest, 32b datacopy, fast_tilize), and a RISC cfg_read is not ordered behind queued CFG writes.
// Only written through LLK_ASSERT_BLOCK and read inside LLK_ASSERT. Without ENABLE_LLK_ASSERT it is declared
// but never defined: those macros keep it in an unevaluated operand, so any other use fails to link.
#ifdef ENABLE_LLK_ASSERT
inline bool llk_assert_fp32_dest_acc_en = false;
#else
extern bool llk_assert_fp32_dest_acc_en;
#endif

// Asserts that a passed is_fp32_dest_acc_en matches the configured dest accumulation mode.
#define LLK_ASSERT_FP32_DEST_ACC_EN_MATCHES(is_fp32_dest_acc_en) \
    LLK_ASSERT(((is_fp32_dest_acc_en) == llk_assert_fp32_dest_acc_en), "is_fp32_dest_acc_en does not match the configured dest accumulation mode")
