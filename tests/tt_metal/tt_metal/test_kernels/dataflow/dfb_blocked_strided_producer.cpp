// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) BLOCKED-producer -> STRIDED-consumer DFB producer.
// A STRIDED consumer owns interleaved ring slots, so tiles have to go over one at a time for
// the round-robin to credit the right consumer. This keeps dfb_blocked_producer's contiguous
// DRAM read order but pushes per tile. A 1x1 ring has nothing to interleave with, so there the
// whole block still moves in one transaction.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    const uint32_t num_blocks = num_entries_per_producer / block_size;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        // This thread's b-th block: block_size contiguous pages, blocks interleaved across producers.
        const uint32_t block_base_page = chunk_offset + (b * num_producers + producer_idx) * block_size;
        // 1 while the ring is interleaved, block_size on a 1x1 ring where it isn't.
#ifdef ARCH_QUASAR
        const uint32_t entries_per_txn = dfb.get_entries_per_txn();
#else
        const uint32_t entries_per_txn = 1;
#endif
        for (uint32_t j = 0; j < block_size; j += entries_per_txn) {
            const uint32_t page_id = block_base_page + j;
            if (page_id >= chunk_offset + entries_per_core) {
                break;
            }
            if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
                noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = page_id}, {});
#endif
            } else {
                dfb.reserve_back(static_cast<uint16_t>(entries_per_txn));
                noc.async_read(tensor_accessor, dfb, entries_per_txn * entry_size, {.page_id = page_id}, {});
                noc.async_read_barrier();
                dfb.push_back(static_cast<uint16_t>(entries_per_txn));
            }
        }
    }
    dfb.finish();
}
