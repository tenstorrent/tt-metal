// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ENV_LLK_INFRA

// Assume we are executing in tt-metal and we have assert already available.
#include "debug/assert.h"

#define LLK_ASSERT(condition, message) ASSERT(condition)

#else

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

#endif
