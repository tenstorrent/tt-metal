// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) D1 DM producer.
// Variant of dfb_producer.cpp that pre-increments the TC posted counter to a
// near-wrap value before the main producer loop.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/dataflow_buffer_test_helpers.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t kPreloadPostedValue = get_arg(args::kPreloadPostedValue);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    if constexpr (kPreloadPostedValue > 0) {
#ifdef ARCH_QUASAR
        if (producer_idx == 0) {
            preload_posted_counter(dfb, kPreloadPostedValue);
            // Cross-kernel rendezvous: this kernel preloads POSTED; the consumer kernel
            // preloads ACKED. They run on separate DM RISCs with no implicit ordering, so
            // without a barrier this producer could start issuing async_reads (advancing
            // posted) before the consumer has bumped acked — leaving occupancy != 0 and
            // corrupting the first entry. Signal "producer ready", then wait for the
            // consumer's "ready" so neither side enters its data loop until both counters
            // are preloaded (occupancy == 0).
            Semaphore prod_ready(sem::prod_ready);
            Semaphore cons_ready(sem::cons_ready);
            prod_ready.set(1);
            cons_ready.wait_min(1);
        }
#endif
    }

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = page_id}, {});
#endif
        } else {
            dfb.reserve_back(1);
            noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    }
    dfb.finish();
}
