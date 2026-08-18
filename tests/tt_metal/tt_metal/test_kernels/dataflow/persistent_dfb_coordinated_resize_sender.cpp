// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Long-running PersistentDFB sender for coordinated live-peer E1→E2.
//
// Fixed safe-point protocol (drain real E1 traffic before resize):
//   1. Construct at E1
//   2. Push num_entries_e1 at E1
//   3. barrier() so acked == sent on all receivers (drain)
//   4. set_entry_size(E2) + internal barrier_sender_credits (NOC credit fixup)
//   5. Signal host via resized_sem
//   6. Wait on go_sem (host launches receiver C while this kernel stays alive)
//   7. Push num_entries_e2 at E2
//
// Compile-time args:
//   [0] persistent_dfb_id
//   [1] entry_size_e1
//   [2] num_entries_e1
//   [3] entry_size_e2
//   [4] num_entries_e2
//   [5] data_pattern      - must be multicast counter (0) for write_broadcast
//
// Runtime args:
//   [0] l1_staging_addr
//   [1] resized_sem_addr  - written to 1 after resize completes
//   [2] go_sem_addr       - wait until host writes 1
//
// Staging layout (host MulticastCounter with resized tail):
//   [0, num_entries_e1 * entry_size_e1)           E1 entries
//   [num_entries_e1 * entry_size_e1, ...)         E2 entries

#include "api/dataflow/persistent_dfb.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint8_t persistent_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size_e1 = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries_e1 = get_compile_time_arg_val(2);
    constexpr uint32_t entry_size_e2 = get_compile_time_arg_val(3);
    constexpr uint32_t num_entries_e2 = get_compile_time_arg_val(4);
    constexpr uint32_t data_pattern = get_compile_time_arg_val(5);

    constexpr uint32_t pattern_multicast_counter = 0;
    static_assert(data_pattern == pattern_multicast_counter, "coordinated resize sender expects multicast counter");
    static_assert(num_entries_e1 > 0, "E1 phase must push at least one entry before resize");

    const uint32_t staging_base = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* resized_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));

    Noc noc;
    experimental::PersistentDFB gdfb(persistent_dfb_id);

    for (uint32_t i = 0; i < num_entries_e1; ++i) {
        gdfb.reserve_back(1);
        gdfb.write_broadcast(staging_base + i * entry_size_e1, 1, noc);
        gdfb.flush_writes(noc);
        gdfb.push_back(1, noc);
    }

    // Author-defined safe point: drain so no in-flight work crosses the page-size change.
    gdfb.barrier();
    gdfb.set_entry_size(entry_size_e2);

    noc_semaphore_set(resized_sem, 1);
    noc_semaphore_wait(go_sem, 1);

    const uint32_t staging_e2 = staging_base + num_entries_e1 * entry_size_e1;
    for (uint32_t i = 0; i < num_entries_e2; ++i) {
        gdfb.reserve_back(1);
        gdfb.write_broadcast(staging_e2 + i * entry_size_e2, 1, noc);
        gdfb.flush_writes(noc);
        gdfb.push_back(1, noc);
    }
}
