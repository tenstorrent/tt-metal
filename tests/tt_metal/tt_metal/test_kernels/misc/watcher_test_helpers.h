// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_QUASAR) || !defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "hostdev/dev_msgs.h"
#include "internal/risc_attribs.h"
#if defined(ARCH_QUASAR)
#include "internal/hw_thread.h"
#else
#include "internal/firmware_common.h"
#endif

// Lets the core be reported done while the kernel deliberately hangs (e.g. on a watcher error), so dispatch can
// drain. Only valid if the kernel never returns.
// On Quasar, user DMs and TRISCs are subordinates of DM0, whose firmware reports the core done for the active
// dispatch mode once every subordinate is done, so this only marks the caller done. Its DM slots start at DM1.
// Elsewhere this notifies the fast dispatcher over the NOC, once per call.
FORCE_INLINE void signal_completion_before_hang() {
#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_TRISC)
    GET_MAILBOX_ADDRESS_DEV(subordinate_sync)->map[internal_::get_hw_thread_idx()] = RUN_SYNC_MSG_DONE;
#else
    GET_MAILBOX_ADDRESS_DEV(subordinate_sync)->map[internal_::get_hw_thread_idx() - 1] = RUN_SYNC_MSG_DONE;
#endif
#else
    const uint32_t go_message_index = *GET_MAILBOX_ADDRESS_DEV(go_message_index);
    uint64_t dispatch_addr = calculate_dispatch_addr(GET_MAILBOX_ADDRESS_DEV(go_messages[go_message_index]));
    notify_dispatch_core_done(dispatch_addr, noc_index);
#endif
}
#endif
