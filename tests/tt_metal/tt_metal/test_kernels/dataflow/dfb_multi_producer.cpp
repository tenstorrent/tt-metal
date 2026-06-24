// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-DFB producer kernel for concurrent DFB stress tests (Gap 7).
//
// Each instance of this kernel runs as a single DM thread and handles exactly one
// DFB.  The DFB is identified by the kernel's `dfb::out` binding (one binding per
// kernel instance).  The harness creates one kernel instance per DFB so each DFB
// gets precisely one producer thread - matching num_producers=1 in the DFB config.
//
// Named args (CTAs):
//   args::num_entries_per_producer
//   args::implicit_sync
//   args::chunk_offset
// Bindings:
//   dfb::out               - producer endpoint of this instance's DFB
//   tensor::src_tensor     - shared DRAM in_tensor accessor

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t chunk_offset = get_arg(args::chunk_offset);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        const uint32_t page_id = chunk_offset + tile_id;
        if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
            noc.async_read<NocOptions::TXN_ID>(
                tensor_accessor, dfb, {.page_id = page_id}, {});
#endif
        } else {
            dfb.reserve_back(1);
            noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    }
    DPRINT("producer before finish\n");
    dfb.finish();
    DPRINT("producer after finish\n");
}
