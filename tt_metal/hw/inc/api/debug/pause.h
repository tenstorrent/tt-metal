// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"
#include "api/debug/waypoint.h"
#include "api/debug/pause.h"
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_PAUSE) && !defined(FORCE_WATCHER_OFF)

void watcher_pause() {
    // Tell the host we're paused and which value of our unpause bit we're waiting to see change (debug_pause_msg_t).
    debug_pause_msg_t tt_l1_ptr* pause_msg = GET_MAILBOX_ADDRESS_DEV(watcher.pause_status);
    const uint32_t idx = internal_::get_hw_thread_idx();
    const uint32_t bit = 1u << idx;
    invalidate_l1_cache();
    const uint32_t unpause = pause_msg->unpause & bit;
    pause_msg->flags[idx] = unpause ? 2 : 1;

    // Wait for the host to flip our bit.
    WAYPOINT("PASW");
    do {
        invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
    } while ((pause_msg->unpause & bit) == unpause);
    pause_msg->flags[idx] = 0;
    WAYPOINT("PASD");
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define PAUSE()          \
    do {                 \
        watcher_pause(); \
    } while (0)

#else  // !WATCHER_ENABLED

#define PAUSE()

#endif  // WATCHER_ENABLED
