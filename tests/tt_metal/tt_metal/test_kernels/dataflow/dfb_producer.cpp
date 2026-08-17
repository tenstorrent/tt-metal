// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t num_producers = get_arg(args::num_producers);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);
    const uint32_t producer_idx = get_my_thread_id();

    DataflowBuffer dfb(dfb::out);
    Noc noc;

    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        // Strided access: producer i owns pages i, i+P, i+2P, ...
        const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
        // Skip if this producer's slice overshoots the actual buffer size (happens when
        // entries_per_core is not a multiple of num_producers).
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        // DPRINT("producer tile id {} page id {}\n", tile_id, page_id);
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
    // DPRINT("PFW\n");
    dfb.finish();
    // DPRINT("PFD\n");
}
