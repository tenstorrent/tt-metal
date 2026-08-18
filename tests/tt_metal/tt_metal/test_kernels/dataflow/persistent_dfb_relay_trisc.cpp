// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TRISC consumer for a PersistentDFB relay DFB.
//
// The RelayDFBBindingToken carries the persistent slot id, so constructing the
// DataflowBuffer snaps the borrowed local iface to the durable fifo_ptr checkpoint
// (O(1) launch-msg slot lookup — no JIT-emitted trisck align).
//
// Compile-time parameters:
//   [0] relay_dfb_id
//   [1] total_entries
//   [2] batch_size
//   [3] delay_iterations
//   [4] persistent_dfb_id
//
// Runtime args:
//   [0] result_l1_addr: [entries_consumed, checksum]

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr uint16_t relay_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_entries = get_compile_time_arg_val(1);
    constexpr uint16_t batch_size = get_compile_time_arg_val(2);
    constexpr uint32_t delay_iterations = get_compile_time_arg_val(3);
    constexpr uint8_t persistent_dfb_id = get_compile_time_arg_val(4);

    DataflowBuffer relay(RelayDFBBindingToken{relay_dfb_id, persistent_dfb_id});
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));

    uint32_t checksum = 0;
    for (uint32_t offset = 0; offset < total_entries; offset += batch_size) {
        relay.wait_front(batch_size);
        const uint32_t read_ptr = relay.get_read_ptr() << cb_addr_shift;
        const uint32_t entry_size = relay.get_entry_size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            checksum += *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_ptr + i * entry_size);
        }
        for (volatile uint32_t delay = 0; delay < delay_iterations; ++delay) {
        }
        relay.pop_front(batch_size);
    }
    result[0] = total_entries;
    result[1] = checksum;
#endif
}
