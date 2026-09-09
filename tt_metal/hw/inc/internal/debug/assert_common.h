// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdev/dev_msgs.h"  // debug_assert_type_t
#include "internal/risc_attribs.h"

// The ASSERT macro alone, so internal/hw_thread.h can use it without an include cycle.
// Include api/debug/assert.h instead of this one: with only this header in scope, ASSERT links
// against an undefined assert_and_hang.

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

// Defined in api/debug/assert.h, after risc_common.h provides flush_l2_cache_range and related symbols.
inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type = DebugAssertTripped);

#define ASSERT(condition, ...) (void(not(condition) ? assert_and_hang(__LINE__, ##__VA_ARGS__), 0 : 0))

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#elif defined(LIGHTWEIGHT_KERNEL_ASSERTS)

// Trap wrapped as a function to avoid inline assembly at ASSERT macro's use site.
FORCE_INLINE void lightweight_assert_trap() { asm("ebreak"); }

#define ASSERT(condition, ...) (void(not(condition) ? lightweight_assert_trap(), 0 : 0))

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#else

// Avoid unused variable warnings here.
#define ASSERT(condition, ...) (void(sizeof(not(condition))))

#define ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0

#endif
