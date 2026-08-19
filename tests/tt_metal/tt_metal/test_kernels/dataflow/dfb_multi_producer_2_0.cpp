// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) multi-DFB producer for A2 concurrent-DFBs tests.
// Each thread instance reads from its own DRAM region and pushes to its own DFB.
// The DFB binding for each thread is selected by host KernelSpec; this kernel
// uses dfb::out (which the host binds to a different DFB per kernel instance).

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);
    const uint32_t entry_size = dfb.get_entry_size();

    for (uint32_t i = 0; i < num_entries_per_producer; ++i) {
        const uint32_t page_id = chunk_offset + i;
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
