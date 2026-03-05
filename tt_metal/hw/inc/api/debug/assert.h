// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

inline void assert_and_hang(
    uint32_t line_num, uint16_t file_id, uint16_t extra_info, debug_assert_type_t assert_type = DebugAssertTripped) {
    // Write the assert info into the memory mailbox for host to read.
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->file_id = file_id;
        v->extra_info = extra_info;
        v->tripped = assert_type;
        v->which = internal_::get_hw_thread_idx();
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

// Backward-compatible overload for existing direct callers.
inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type = DebugAssertTripped) {
    assert_and_hang(line_num, 0, 0, assert_type);
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define ASSERT(condition, ...)                                                      \
    do {                                                                            \
        if (not(condition))                                                         \
            assert_and_hang(__LINE__, debug_file_hash(__FILE__), 0, ##__VA_ARGS__); \
    } while (0)

#define ASSERT_MSG(condition, message)                                                     \
    do {                                                                                   \
        if (not(condition))                                                                \
            assert_and_hang(__LINE__, debug_file_hash(__FILE__), debug_msg_hash(message)); \
    } while (0)

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#else  // !WATCHER_ENABLED

#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)

#define ASSERT(condition, ...)      \
    do {                            \
        if (!(condition))           \
            asm volatile("ebreak"); \
    } while (0)

#define ASSERT_MSG(condition, message) ASSERT(condition)

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#else  // !LIGHTWEIGHT_KERNEL_ASSERTS

#define ASSERT(condition, ...)
#define ASSERT_MSG(condition, message)

#define ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#endif  // !LIGHTWEIGHT_KERNEL_ASSERTS

#endif  // WATCHER_ENABLED
