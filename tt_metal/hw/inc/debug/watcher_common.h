// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_msgs.h"

#if defined(WATCHER_ENABLED)

#if defined(COMPILE_FOR_ERISC)
// Forward declare these functions to avoid including erisc.h -> circular dependency via noc_nonblocking_api.h, sanitize_noc.h.
namespace internal_ {
void __attribute__((section("code_l1"))) risc_context_switch();
void disable_erisc_app();
}

// Pointer to exit routine, (so it may be called from a kernel).
[[gnu::noreturn]] extern void (*erisc_exit)();

#endif

inline uint32_t debug_get_which_riscv()
{
#if defined(COMPILE_FOR_BRISC)
    return DebugBrisc;
#elif defined(COMPILE_FOR_NCRISC)
    return DebugNCrisc;
#elif defined(COMPILE_FOR_ERISC)
    return DebugErisc;
#elif defined(COMPILE_FOR_IDLE_ERISC)
    return DebugIErisc;
#else
    return DebugTrisc0 + COMPILE_FOR_TRISC;
#endif
}

void clear_previous_launch_message_entry_for_watcher() {
    uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    // Once the read pointer has been incremented, clear the watcher info 2 entries before to ensure that we don't report stale data
    uint32_t prev_rd_ptr = (launch_msg_rd_ptr - 2 + launch_msg_buffer_num_entries) % launch_msg_buffer_num_entries;
    launch_msg_t tt_l1_ptr *launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[prev_rd_ptr]);
    // Clear kernel ids and NOC ID used by stale program entry, since these are queried by watcher
    for (int idx = 0; idx < DISPATCH_CLASS_MAX; idx++) {
        launch_msg->kernel_config.watcher_kernel_ids[idx] = 0;
    }
    launch_msg->kernel_config.brisc_noc_id = 0;
}
#define CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER() clear_previous_launch_message_entry_for_watcher()

#else // !WATCHER_ENABLED

#define CLEAR_PREVIOUS_LAUNCH_MESSAGE_ENTRY_FOR_WATCHER()

#endif // WATCHER_ENABLED
