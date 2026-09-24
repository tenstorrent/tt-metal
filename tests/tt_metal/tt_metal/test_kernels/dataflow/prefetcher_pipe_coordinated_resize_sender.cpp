// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Long-running PrefetcherPipe sender for coordinated live-peer E1→E2.
//
// Mixed-size prefetch protocol:
//   1. Construct at E1
//   2. Push num_entries_e1 at E1
//   3. set_entry_size(E2), without waiting for the E1 receiver to drain
//   4. Signal host via resized_sem
//   5. Wait on go_sem (host launches receiver C while this kernel stays alive)
//   6. Push num_entries_e2 at E2
//
// Bindings:
//   pipe::out              — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::entry_size_e1
//   args::num_entries_e1
//   args::entry_size_e2
//   args::num_entries_e2
// Args (named RTAs):
//   args::staging_addr      - multicast-counter staging (see layout below)
//   args::resized_sem_addr  - written to 1 after resize completes
//   args::go_sem_addr       - wait until host writes 1
//   args::credit_base_addr  - Quasar only: pages_sent slot for host credit probe flush
//
// Staging layout (host MulticastCounter with resized tail):
//   [0, num_entries_e1 * entry_size_e1)           E1 entries
//   [num_entries_e1 * entry_size_e1, ...)         E2 entries

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#ifdef ARCH_QUASAR
#include "dev_mem_map.h"
#endif

void kernel_main() {
    constexpr uint32_t entry_size_e1 = get_arg(args::entry_size_e1);
    constexpr uint32_t num_entries_e1 = get_arg(args::num_entries_e1);
    constexpr uint32_t entry_size_e2 = get_arg(args::entry_size_e2);
    constexpr uint32_t num_entries_e2 = get_arg(args::num_entries_e2);

    static_assert(num_entries_e1 > 0, "E1 phase must push at least one entry before resize");

    const uint32_t staging_base = get_arg(args::staging_addr);
    const CoreLocalMem<uint8_t> staging(staging_base);
    // Quasar: host read_core/write_core see the uncached L1 alias. Cached stores are not
    // host-visible without a flush (same pattern as sub_device/syncer.cpp).
#ifdef ARCH_QUASAR
    volatile tt_l1_ptr uint32_t* resized_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::resized_sem_addr) + MEM_L1_UNCACHED_BASE);
    volatile tt_l1_ptr uint32_t* go_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::go_sem_addr) + MEM_L1_UNCACHED_BASE);
    const uint32_t credit_base_addr = get_arg(args::credit_base_addr);
#else
    volatile tt_l1_ptr uint32_t* resized_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::resized_sem_addr));
    volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::go_sem_addr));
#endif

    Noc noc;
    experimental::PrefetcherPipe gdfb(pipe::out);

    for (uint32_t i = 0; i < num_entries_e1; ++i) {
        gdfb.reserve_back(1);
        gdfb.write_broadcast(noc, staging, 1, {.offset_bytes = i * entry_size_e1});
        gdfb.flush_writes(noc);
        gdfb.push_back(1, noc);
    }

    gdfb.set_entry_size(entry_size_e2);

#ifdef ARCH_QUASAR
    // pages_sent / wr_offset are updated via cached stores; host read_credit_pair reads TL1.
    flush_l2_cache_range(credit_base_addr, 2 * L1_ALIGNMENT);
#endif
    noc_semaphore_set(resized_sem, 1);
    noc_semaphore_wait(go_sem, 1);

    const uint32_t staging_e2_offset = num_entries_e1 * entry_size_e1;
    for (uint32_t i = 0; i < num_entries_e2; ++i) {
        gdfb.reserve_back(1);
        gdfb.write_broadcast(noc, staging, 1, {.offset_bytes = staging_e2_offset + i * entry_size_e2});
        gdfb.flush_writes(noc);
        gdfb.push_back(1, noc);
    }
}
