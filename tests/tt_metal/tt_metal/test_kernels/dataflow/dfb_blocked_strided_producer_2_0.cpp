// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) BLOCKED-producer -> STRIDED-consumer DFB producer.
//
// Unlike dfb_blocked_producer_2_0.cpp (which bursts a whole block to one contiguous
// sub-ring), a STRIDED consumer reads INTERLEAVED ring slots {c, c+N, c+2N, ...}, so the
// producer must hand its tiles to the consumers ONE AT A TIME: the DFB's STRIDED round-robin
// then posts each tile's credit to the next consumer's tile-counter and advances that
// consumer's interleaved write pointer. So this kernel keeps the BLOCKED *DRAM read order*
// (block_size contiguous pages per block, blocks interleaved across producers) but PUSHES
// PER-TILE. No remapper, no broadcast, no credit-path change — it reuses the verbatim STRIDED
// round-robin (the producer's num_tcs_to_rr == fan-out width).
//
// Bindings/CTAs (set by host KernelSpec):
//   dfb::out                  — PRODUCER
//   num_entries_per_producer  — total entries this thread produces
//   block_size                — tiles per DRAM block
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
        // BLOCKED DRAM read order: this thread's b-th block is a contiguous run of block_size
        // pages, blocks interleaved across producers by block_size * num_producers.
        const uint32_t block_base_page = chunk_offset + (b * num_producers + producer_idx) * block_size;
        // STRIDED delivery: push the block's tiles one at a time so the round-robin scatters them
        // across the consumers' interleaved ring slots (a block burst would land in one consumer).
        for (uint32_t j = 0; j < block_size; ++j) {
            const uint32_t page_id = block_base_page + j;
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
    }
    dfb.finish();
}
