// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Legacy-API DM kernel for SentinelPatternHandlingAndMissingRTADetection (Part B).
// Signals dispatcher completion, then reads RTA[0]. On cores where the host never called
// SetRuntimeArgs, the dispatcher leaves the 0xBEEF#### sentinel pattern; the device
// interprets it as rta_count = 0, so the RTA[0] access is out-of-bounds and trips the
// watcher assert. This scenario can't be expressed via the Metal 2.0 host API because
// SetProgramRunArgs validates that every targeted node has runtime args bound.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Signal dispatcher completion before the OOB access so FD can drain.
    volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
    notify_dispatch_core_done(dispatch_addr, noc_index);

    volatile uint32_t rta = get_arg_val<uint32_t>(0);
    (void)rta;
}
