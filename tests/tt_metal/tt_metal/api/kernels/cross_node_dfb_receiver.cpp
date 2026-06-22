// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CrossNodeDFB receiver kernel: wait for entries and pop
//
// Compile-time parameters (via kernel compile_args):
//   [0] remote_dfb_id
//   [1] entry_size
//   [2] num_entries        - entries to pop before optional resize
//   [3] receiver_idx       - unused (reserved for test harness symmetry)
//   [4] do_commit          - 1 to call commit() at end
//   [5] offset_entries     - unused (host verifies expected data layout)
//   [6] do_resize          - 1 to resize after initial entries
//   [7] entry_size_resized - new entry_size after resize
//   [8] num_entries_after  - entries to pop after resize
//   [9] relay_cb_id        - 0xFF to disable relay DFB registration in-kernel

#include "api/dataflow/cross_node_dfb.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t  remote_dfb_id      = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size         = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries        = get_compile_time_arg_val(2);
    constexpr uint32_t do_commit          = get_compile_time_arg_val(4);
    constexpr uint32_t do_resize          = get_compile_time_arg_val(6);
    constexpr uint32_t entry_size_resized = get_compile_time_arg_val(7);
    constexpr uint32_t num_entries_after  = get_compile_time_arg_val(8);

    Noc noc;

    experimental::CrossNodeDFB gdfb(remote_dfb_id);

    for (uint32_t i = 0; i < num_entries; ++i) {
        DPRINT("Doing wait front\n");
        gdfb.wait_front(1);
        DPRINT("Done wait front\n");
        gdfb.pop_front(1, noc);
        DPRINT("Done pop front\n");
    }

    if constexpr (do_resize) {
        gdfb.set_receiver_entry_size(entry_size_resized, noc);
        for (uint32_t i = 0; i < num_entries_after; ++i) {
            gdfb.wait_front(1);
            gdfb.pop_front(1, noc);
        }
    }

    if constexpr (do_commit) {
        DPRINT("Doing commit\n");
        gdfb.commit();
        DPRINT("Done commit\n");
    }
}
