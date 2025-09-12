// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"
#include "waypoint.h"
#include "debug/pause.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_PAUSE) && !defined(FORCE_WATCHER_OFF)

void watcher_pause() {
    // Write the pause flag for this core into the memory mailbox for host to read.
    debug_pause_msg_t tt_l1_ptr* pause_msg = GET_MAILBOX_ADDRESS_DEV(watcher.pause_status);
    pause_msg->flags[PROCESSOR_INDEX] = 1;

    // Wait for the pause flag to be cleared.
    WAYPOINT("PASW");
    do {
        invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
    } while (pause_msg->flags[PROCESSOR_INDEX]);
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
