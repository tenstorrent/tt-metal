// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Simple DM producer kernel for borrowed-memory DFB tests.
// Reads from a pre-filled L1 input region into the DFB ring (which is the borrowed L1 buffer).
//
// Compile-time args:
//   CTA[0]: num_entries_per_risc  - number of entries this RISC produces
//   CTA[1]: entry_size            - size of each entry in bytes
//
// Runtime args:
//   RTA[0]: l1_src_base_addr      - base address of L1 input region (host pre-fills this)

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    const uint32_t l1_src_base = get_arg_val<uint32_t>(0);

    const uint32_t risc_idx = get_my_thread_id();
    const uint32_t src_base = l1_src_base + risc_idx * num_entries * entry_size;

    DataflowBuffer dfb(0);

    for (uint32_t i = 0; i < num_entries; i++) {
        dfb.reserve_back(1);
        const uint64_t src_noc = get_noc_addr(src_base + i * entry_size);
        noc_async_read(src_noc, dfb.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        dfb.push_back(1);
    }

    dfb.finish();
}
