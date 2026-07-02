// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CrossNodeDFB DM receiver with relay DFB bridging to TRISC.
//
// Compile-time parameters:
//   [0] remote_dfb_id
//   [1] entry_size
//   [2] num_entries
//   [3] relay_cb_id

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/cross_node_dfb.h"
#include "api/dataflow/noc.h"

struct RelayCB {
    uint32_t id;
    explicit RelayCB(uint32_t cb_id) : id(cb_id) {}
    uint8_t get_logical_handle() const { return static_cast<uint8_t>(id); }
};

void kernel_main() {
    constexpr uint8_t remote_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);
    constexpr uint32_t relay_cb_id = get_compile_time_arg_val(3);

    Noc noc;
    experimental::CrossNodeDFB gdfb(remote_dfb_id);
    CircularBuffer relay(relay_cb_id);
    RelayCB relay_handle(relay_cb_id);
    gdfb.register_relay_dfbs(relay_handle);

    for (uint32_t i = 0; i < num_entries; ++i) {
        relay.reserve_back(1);
        gdfb.wait_front(1);
        gdfb.push_relay_front(1);
        gdfb.pop_front(1, noc);
    }

    gdfb.commit();
}
