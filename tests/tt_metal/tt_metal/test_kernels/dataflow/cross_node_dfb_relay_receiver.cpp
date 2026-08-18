// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CrossNodeDFB DM receiver with relay DFB bridging to TRISC.
//
// Compile-time parameters:
//   [0] remote_dfb_id
//   [1] total_entries
//   [2] batch_size

#include "api/dataflow/cross_node_dfb.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t remote_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_entries = get_compile_time_arg_val(1);
    constexpr uint16_t batch_size = get_compile_time_arg_val(2);

    Noc noc;
    experimental::CrossNodeDFB cn_dfb(remote_dfb_id);
    auto relay = cn_dfb.bind_relay();

    for (uint32_t offset = 0; offset < total_entries; offset += batch_size) {
        relay.reserve_back(batch_size);
        cn_dfb.wait_front(batch_size);
        relay.push_back(batch_size);

        // TRISC pop is the lifetime boundary for the aliased L1 entries.
        relay.wait_consumed(batch_size);
        cn_dfb.pop_front(batch_size, noc);
    }
}
