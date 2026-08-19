// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver half of a coordinated PersistentDFB page-size change:
// consume all E1 traffic, then resize to E2 and consume the sender's pad
// credits. The sender may already be configured for E2 while E1 is consumed.
//
// Compile-time args:
//   [0] persistent_dfb_id
//   [1] num_entries_e1
//   [2] entry_size_e2

#include "api/dataflow/persistent_dfb.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t persistent_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries_e1 = get_compile_time_arg_val(1);
    constexpr uint32_t entry_size_e2 = get_compile_time_arg_val(2);

    Noc noc;
    experimental::PersistentDFB dfb(persistent_dfb_id);
    for (uint32_t i = 0; i < num_entries_e1; ++i) {
        dfb.wait_front(1);
        dfb.pop_front(1, noc);
    }
    dfb.set_receiver_entry_size(entry_size_e2);
}
