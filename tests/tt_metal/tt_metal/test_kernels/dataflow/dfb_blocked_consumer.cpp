// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) BLOCKED DFB consumer.
// Parallel to dfb_consumer_2_0.cpp, but waits on and drains block_size contiguous entries
// per NoC transaction, then strides by block_size * num_consumers.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::in);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::dst_tensor);

    const uint32_t consumer_idx = get_my_thread_id();
    const uint32_t num_consumers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    const uint32_t num_blocks = num_entries_per_consumer / block_size;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        // This thread's b-th block: block_size contiguous pages, blocks interleaved across consumers.
        const uint32_t block_base_page = chunk_offset + (b * num_consumers + consumer_idx) * block_size;
        if (block_base_page >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            // This kernel IS the blocked side, so one transaction drains the whole block; the
            // library is told rather than left to guess. Don't pass an explicit byte size --
            // that picks the generic Noc::async_write, which posts no credits.
            noc.async_write<NocOptions::TXN_ID>(
                dfb, tensor_accessor, {.num_entries = block_size}, {.page_id = block_base_page});
#endif
        } else {
            dfb.wait_front(block_size);
            // One transaction for the whole block; the read pointer is already the block base.
            noc.async_write(dfb, tensor_accessor, block_size * entry_size, {}, {.page_id = block_base_page});
            noc.async_write_barrier();
            dfb.pop_front(block_size);
        }
    }
    dfb.finish();
    dfb.write_barrier(noc);
}
