// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"
#include "eth_fw_api.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ETH_LINK_STATUS) && !defined(FORCE_WATCHER_OFF) && \
    defined(COMPILE_FOR_ERISC)

void hang_on_down_link() {
    debug_eth_link_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.eth_status);
    v->link_down = 1;

    // Update launch msg to show that we've exited. This is required so that the next run doesn't think there's a kernel
    // still running and try to make it exit.
    volatile tt_l1_ptr go_msg_t* go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_ptr->signal = RUN_MSG_DONE;

    // This exits to base FW
    internal_::disable_erisc_app();
    erisc_exit();

    while (1) {
        ;
    }
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define WATCHER_CHECK_ETH_LINK_STATUS() \
    do {                                \
        if (not(is_link_up()))          \
            hang_on_down_link();        \
    } while (0)

#define WATCHER_ETH_LINK_STATUS_ENABLED 1

#else  // !WATCHER_ENABLED or !COMPILE_FOR_ERISC

#define WATCHER_CHECK_ETH_LINK_STATUS()

#define WATCHER_ETH_LINK_STATUS_ENABLED 0

#endif  // WATCHER_ENABLED
