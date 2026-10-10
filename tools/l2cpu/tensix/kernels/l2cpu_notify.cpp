// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Notify: req_seq += 1 in the link channel (after this core's earlier writes are acked), then the doorbell.
// Runtime args: l2_x, l2_y, base_hi, base_lo, diag. diag != 0: the wall clock after the doorbell goes to
// DIAG line 0 word 0 (read by l2cpu_wait.cpp in diag mode). CB 0: >= 1 KiB L1 scratch.
#include "api/dataflow/dataflow_api.h"
#include "l2cpu_noc.h"

void kernel_main() {
    l2cpu_link c{
        noc_index,
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        ((uint64_t)get_arg_val<uint32_t>(2) << 32) | get_arg_val<uint32_t>(3),
        get_write_ptr(0)};
    l2cpu_link_notify(c);
    if (get_arg_val<uint32_t>(4)) {
        l2cpu_link_write32(c, L2CPU_LINK_OFF_DIAG, get_timestamp_32b(), 3);
    }
}
