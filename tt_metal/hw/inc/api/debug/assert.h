// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"

// ---------------------------------------------------------------------------
// Compile-time FNV-1a file hash
//
// Produces a stable uint16_t "file ID" from __FILE__ at compile time.
// The host reconstructs the same hash over every known kernel source path to
// resolve the file name from the ID stored in the assert mailbox.
// Using the low 16 bits of a 32-bit FNV-1a hash keeps the mailbox impact to
// 2 bytes while being collision-resistant enough for typical kernel source
// sets (probability of any collision across ~1000 files is < 0.75%).
// ---------------------------------------------------------------------------
constexpr uint16_t watcher_file_hash(const char* s, uint32_t h = 2166136261u) {
    return (*s == '\0') ? static_cast<uint16_t>(h & 0xFFFFu)
                        : watcher_file_hash(s + 1, (h ^ static_cast<uint8_t>(*s)) * 16777619u);
}

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

// file_id: low 16 bits of FNV-1a hash of __FILE__ — pass watcher_file_hash(__FILE__).
// assert_type: defaults to DebugAssertTripped for the general ASSERT() macro; other
// assert types (NOC races, RTA out-of-bounds, etc.) pass their own type and a
// file_id of 0 (they don't originate from a kernel ASSERT() call site).
inline void assert_and_hang(
    uint32_t line_num, uint16_t file_id = 0, debug_assert_type_t assert_type = DebugAssertTripped) {
    // Write the line number into the memory mailbox for host to read.
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->tripped = assert_type;
        v->which = internal_::get_hw_thread_idx();
        v->file_id = file_id;
    }

    // Hang, or in the case of erisc, early exit.
#if defined(COMPILE_FOR_ERISC)
    // Update launch msg to show that we've exited. This is required so that the next run doesn't think there's a kernel
    // still running and try to make it exit.
    volatile tt_l1_ptr go_msg_t* go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_ptr->signal = RUN_MSG_DONE;

    // This exits to base FW
    internal_::disable_erisc_app();
    // Subordinates do not have an erisc exit
#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
    erisc_exit();
#endif
#endif

    while (1) {
        ;
    }
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define ASSERT(condition, ...)                                                     \
    do {                                                                           \
        if (not(condition))                                                        \
            assert_and_hang(__LINE__, watcher_file_hash(__FILE__), ##__VA_ARGS__); \
    } while (0)

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#else  // !WATCHER_ENABLED

#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)

#define ASSERT(condition, ...) \
    do {                       \
        if (!(condition))      \
            while (true);      \
    } while (0)

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#elif defined(ENABLE_LLK_ASSERT)

#define ASSERT(condition, ...)

#define ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0

#else  // No asserts enabled

#define ASSERT(condition, ...)

#define ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#endif  // LIGHTWEIGHT_KERNEL_ASSERTS / ENABLE_LLK_ASSERT

#endif  // WATCHER_ENABLED
