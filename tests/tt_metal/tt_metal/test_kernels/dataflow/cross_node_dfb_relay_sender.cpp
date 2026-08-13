// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CrossNode relay sender. The transfer count is independent of ring depth so
// tests can force multiple wraps and exercise relay backpressure.
//
// Compile-time parameters:
//   [0] remote_dfb_id
//   [1] entry_size
//   [2] total_entries
//   [3] batch_size
//
// Runtime args:
//   [0] sender-local staging address

#include "api/dataflow/cross_node_dfb.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t remote_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    constexpr uint32_t total_entries = get_compile_time_arg_val(2);
    constexpr uint32_t batch_size = get_compile_time_arg_val(3);

    const uint32_t staging_base = get_arg_val<uint32_t>(0);
    Noc noc;
    experimental::CrossNodeDFB cn_dfb(remote_dfb_id);

    for (uint32_t offset = 0; offset < total_entries; offset += batch_size) {
        cn_dfb.reserve_back(batch_size);
        cn_dfb.write_broadcast(staging_base + offset * entry_size, batch_size, noc);
        cn_dfb.flush_writes(noc);
        cn_dfb.push_back(batch_size, noc);
    }
}
