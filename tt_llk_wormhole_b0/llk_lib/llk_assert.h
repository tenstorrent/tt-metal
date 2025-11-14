// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ENV_LLK_INFRA

// Assume we are executing in tt-metal and we have assert already available.
#include "debug/assert.h"

#define LLK_ASSERT(condition, message) ASSERT(condition)

#else

// TODO: implement asserts in LLK infra
#define LLK_ASSERT(condition, message)

#endif
