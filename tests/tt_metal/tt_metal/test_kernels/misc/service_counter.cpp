// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "internal/firmware_common.h"

void kernel_main() {
    uint32_t stop_addr = get_arg_val<uint32_t>(0);
    uint32_t counter_addr = get_arg_val<uint32_t>(1);
    uint32_t service_done_addr = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr);
    volatile tt_l1_ptr uint32_t* service_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(service_done_addr);

    // GET_MAILBOX_ADDRESS_DEV(go_messages[0]) yields a go_msg_t* — the go message
    // for this BRISC core. Used to detect and acknowledge FD dispatch go signals.
    volatile tt_l1_ptr go_msg_t* go_msg = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);

    // Consume the initial SD launch go signal without notifying:
    // there is no FD dispatcher at SD launch time.
    go_msg->signal = RUN_MSG_DONE;

    //     while (*stop == 0) {
    //         (*counter)++;

    //         // When the FD dispatcher multicasts a go signal to this core (as part of any
    //         // EnqueueMeshWorkload), acknowledge it immediately via notify_dispatch_core_done.
    //         // Without this the dispatcher waits forever for a DONE from this core.
    //         if (go_msg->signal == RUN_MSG_GO) {
    // #if defined(COMPILE_FOR_BRISC)
    //             uint64_t dispatch_addr = calculate_dispatch_addr(go_msg);
    //             notify_dispatch_core_done(dispatch_addr, noc_index);
    // #endif
    //             go_msg->signal = RUN_MSG_DONE;
    //         }
    //     }

    // Signal host that the kernel has exited. We use a dedicated L1 word rather than
    // go_msg->signal because that is co-opted above for FD dispatch signaling.
    *service_done = 1;
}
