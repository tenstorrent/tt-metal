// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_LLK_ASSERT

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

#endif // ENABLE_LLK_ASSERT
