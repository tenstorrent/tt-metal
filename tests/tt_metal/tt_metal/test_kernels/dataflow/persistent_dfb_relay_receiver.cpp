// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PersistentDFB DM receiver with relay DFB bridging to TRISC.
//
// Compile-time parameters:
//   [0] persistent_dfb_id
//   [1] total_entries
//   [2] batch_size

#include "api/dataflow/persistent_dfb.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t persistent_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_entries = get_compile_time_arg_val(1);
    constexpr uint16_t batch_size = get_compile_time_arg_val(2);

    Noc noc;
    experimental::PersistentDFB pdfb(persistent_dfb_id);
    auto relay = pdfb.bind_relay();

    for (uint32_t offset = 0; offset < total_entries; offset += batch_size) {
        relay.reserve_back(batch_size);
        pdfb.wait_front(batch_size);
        relay.push_back(batch_size);
        relay.wait_consumed(batch_size);
        pdfb.pop_front(batch_size, noc);
    }
}
