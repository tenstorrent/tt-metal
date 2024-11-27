// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "risc_attribs.h"
#include "dataflow_api.h"

// Macros for determining if an early exit is signalled, ERISC only.
#if defined(COMPILE_FOR_IDLE_ERISC)
// Helper function to determine if the dispatch kernel needs to early exit, only valid for IERISC.
bool early_exit() {
    tt_l1_ptr mailboxes_t* const mailbox = (tt_l1_ptr mailboxes_t*)(MEM_IERISC_MAILBOX_BASE);
    return mailbox->launch[mailbox->launch_msg_rd_ptr].kernel_config.exit_erisc_kernel;
}

#define IDLE_ERISC_RETURN(...) \
    if (early_exit()) {        \
        return __VA_ARGS__;    \
    }

#define IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, ...) \
    RISC_POST_HEARTBEAT(heartbeat);                     \
    IDLE_ERISC_RETURN(__VA_ARGS__);

#else

#define IDLE_ERISC_RETURN(...)
#define IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, ...)

#endif
