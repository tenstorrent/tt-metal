// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) BLOCKED DFB producer.
//
// Parallel to dfb_producer_2_0.cpp, but for the BLOCKED access pattern: each
// thread processes a contiguous block of `block_size` entries at a time, then
// strides by block_size * num_producers to its next block. The whole block is
// moved in a single NoC burst (block_size * entry_size bytes) into the block's
// contiguous L1 region, and credits are posted block-at-a-time.
//
// Bindings/CTAs (set by host KernelSpec):
//   dfb::out                  — PRODUCER
//   num_entries_per_producer  — total entries this thread produces
//   block_size                — tiles per block
//   chunk_offset / entries_per_core (RTAs)

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
    const auto tensor_accessor = TensorAccessor(ta::src_tensor);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    const uint32_t num_blocks = num_entries_per_producer / block_size;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        // This thread's b-th block: contiguous run of block_size pages, blocks
        // interleaved across producers by block_size * num_producers.
        const uint32_t block_base_page = chunk_offset + (b * num_producers + producer_idx) * block_size;
        if (block_base_page >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            // Implicit sync is single-entry (txn-id per tile), so a block can't be one burst —
            // issue one TXN_ID read per tile. The block-ness lives in the page sequence and the
            // per-thread contiguous sub-ring; the ISR batches credits.
            for (uint32_t j = 0; j < block_size; ++j) {
                noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = block_base_page + j}, {});
            }
#endif
        } else {
            dfb.reserve_back(block_size);
            // Single NoC burst: read block_size contiguous DRAM pages into the block's
            // contiguous L1 region (dfb write pointer is the block base).
            noc.async_read(tensor_accessor, dfb, block_size * entry_size, {.page_id = block_base_page}, {});
            noc.async_read_barrier();
            dfb.push_back(block_size);
        }
    }
    dfb.finish();
}
