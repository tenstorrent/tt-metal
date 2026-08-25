// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) D1 DM consumer.
// Variant of dfb_consumer.cpp that pre-increments the TC acked counter.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/dataflow_buffer_test_helpers.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    constexpr uint32_t blocked_consumer = get_arg(args::blocked_consumer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t kPreloadAckedValue = get_arg(args::kPreloadAckedValue);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::in);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::dst_tensor);

    const uint32_t consumer_idx = get_my_thread_id();
    const uint32_t num_consumers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    if constexpr (kPreloadAckedValue > 0) {
#ifdef ARCH_QUASAR
        if (consumer_idx == 0) {
            preload_acked_counter(dfb, kPreloadAckedValue);
            // Cross-kernel rendezvous (see dfb_producer_with_tc_preload_2_0.cpp): signal
            // "consumer ready" then wait for the producer so neither side enters its data
            // loop until both POSTED (producer) and ACKED (here) are preloaded — occupancy
            // is then provably 0 and the first entry is not consumed prematurely.
            Semaphore prod_ready(sem::prod_ready);
            Semaphore cons_ready(sem::cons_ready);
            cons_ready.set(1);
            prod_ready.wait_min(1);
        }
#endif
    }

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; ++tile_id) {
        uint32_t page_id = 0;
        if constexpr (blocked_consumer) {
            page_id = chunk_offset + tile_id;
        } else {
            page_id = chunk_offset + tile_id * num_consumers + consumer_idx;
        }
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_write<NocOptions::TXN_ID>(dfb, tensor_accessor, {}, {.page_id = page_id});
#endif
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    dfb.finish();
    dfb.write_barrier(noc);
}
