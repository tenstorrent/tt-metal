// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB consumer kernel for concurrent DFB stress tests (Gap 7).
//
// Each instance handles exactly one DFB (via the dfb::in binding) and drains
// entries to its chunk of a shared DRAM out_buffer (offset by chunk_offset).
// The harness creates one instance per DFB so each DFB gets exactly one
// consumer thread - matching num_consumers=1 in the DFB config.
//
// Named args (CTAs):
//   args::num_entries_per_consumer
//   args::implicit_sync
//   args::chunk_offset
// Bindings:
//   dfb::in                - consumer endpoint of this instance's DFB
//   tensor::dst_tensor     - shared DRAM out_tensor accessor

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t chunk_offset = get_arg(args::chunk_offset);

    DataflowBuffer dfb(dfb::in);
    Noc noc;
    const uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(tensor::dst_tensor);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        const uint32_t page_id = chunk_offset + tile_id;
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_write<NocOptions::TXN_ID>(
                dfb, tensor_accessor, {}, {.page_id = page_id});
#endif
        } else {
            DPRINT("consumer wait page id: {}\n", page_id);
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    DPRINT("consumer before finish\n");
    dfb.finish();
    DPRINT("consumer after finish before write barrier\n");
    dfb.write_barrier(noc);
    DPRINT("finished write barrier\n");
}
