// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TRISC consumer for a CrossNodeDFB relay CB. Mirrors GlobalCB's
// ALIGN_LOCAL_CBS_TO_REMOTE_CBS setup: align relay CB to the live
// CrossNodeReceiverDFBInterface, then wait/pop entries relayed by DM.
//
// Compile-time parameters:
//   [0] remote_dfb_id
//   [1] relay_cb_id
//   [2] num_entries

#include "api/dataflow/dataflow_api.h"
#include "internal/cross_node_dfb_init.h"

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr uint8_t remote_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t relay_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);

    experimental::align_local_cbs_to_cross_node_receiver_dfb(remote_dfb_id, {relay_cb_id});

    for (uint32_t i = 0; i < num_entries; ++i) {
        cb_wait_front(relay_cb_id, 1);
        cb_pop_front(relay_cb_id, 1);
    }
#endif
}
