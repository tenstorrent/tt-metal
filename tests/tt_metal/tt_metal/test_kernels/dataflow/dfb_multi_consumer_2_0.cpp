// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) multi-DFB consumer for A2 concurrent-DFBs tests.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);

    DataflowBuffer dfb(dfb::in);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::dst_tensor);
    const uint32_t entry_size = dfb.get_entry_size();

    for (uint32_t i = 0; i < num_entries_per_consumer; ++i) {
        const uint32_t page_id = chunk_offset + i;
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
