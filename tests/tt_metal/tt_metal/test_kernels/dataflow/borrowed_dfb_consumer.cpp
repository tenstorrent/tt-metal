// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Simple DM consumer kernel for borrowed-memory DFB tests.
// Reads from the DFB ring (borrowed L1 buffer) and writes to a separate L1 output region.
//
// Compile-time args:
//   CTA[0]: num_entries_per_risc  - number of entries this RISC consumes
//   CTA[1]: entry_size            - size of each entry in bytes
//
// Runtime args:
//   RTA[0]: l1_dst_base_addr      - base address of L1 output region (host reads this back)

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    const uint32_t l1_dst_base = get_arg_val<uint32_t>(0);

    const uint32_t risc_idx = get_my_thread_id();
    const uint32_t dst_base = l1_dst_base + risc_idx * num_entries * entry_size;

    DataflowBuffer dfb(0);

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb.wait_front(1);
        const uint64_t dst_noc = get_noc_addr(dst_base + i * entry_size);
        noc_async_write(dfb.get_read_ptr(), dst_noc, entry_size);
        noc_async_write_barrier();
        dfb.pop_front(1);
    }

    dfb.finish();
}
