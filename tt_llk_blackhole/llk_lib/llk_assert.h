// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_LLK_ASSERT

#ifdef ENV_LLK_INFRA

#include "ckernel.h"

#define LLK_ASSERT(condition, message) \
    do                                 \
    {                                  \
        if (UNLIKELY(!(condition)))    \
        {                              \
            asm volatile("ebreak");    \
            UNREACHABLE();             \
        }                              \
    } while (0)

#else

// Assume we are executing in tt-metal and we have assert already available.
#include "api/debug/assert.h"

#define LLK_ASSERT(condition, message) ASSERT(condition)

#endif // ENV_LLK_INFRA

#else

// sizeof creates an unevaluated context: the condition is fully compiled
// (type-checked, name-resolved) but never executed at runtime.
#define LLK_ASSERT(condition, message) ((void)sizeof((condition)))

#endif // ENABLE_LLK_ASSERT
