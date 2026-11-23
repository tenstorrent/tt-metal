// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host stand-in for tt-llk's common/llk_assert.h, for the host-only tests (tensor_views_test.cpp): a failed
// LLK_ASSERT is counted instead of stopping the core, so a test can see that a run-time check reached it.

#pragma once

inline int llk_assert_failures = 0;

#define LLK_ASSERT(condition, message) ((condition) ? void(0) : void(++llk_assert_failures))
