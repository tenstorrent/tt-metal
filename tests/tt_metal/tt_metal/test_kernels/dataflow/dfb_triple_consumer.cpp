// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

template <bool ImplicitSync, typename Dfb, typename Acc>
static inline void consume_one(
    Dfb& dfb, const Acc& dst, Noc& noc, uint32_t num_entries, uint32_t chunk_offset, uint32_t entries_per_core) {
    const uint32_t consumer_idx = get_my_thread_id();
    const uint32_t num_consumers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();
    for (uint32_t tile = 0; tile < num_entries; ++tile) {
        const uint32_t page_id = chunk_offset + tile * num_consumers + consumer_idx;
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (ImplicitSync) {
#ifdef ARCH_QUASAR
            noc.async_write<NocOptions::TXN_ID>(dfb, dst, {}, {.page_id = page_id});
#endif
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, dst, entry_size, {}, {.page_id = page_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    dfb.finish();
    dfb.write_barrier(noc);
}

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries_per_consumer);
    constexpr bool implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    Noc noc;
    DataflowBuffer dfb_0(dfb::in_0);
    DataflowBuffer dfb_1(dfb::in_1);
    DataflowBuffer dfb_2(dfb::in_2);
    const auto dst_0 = TensorAccessor(tensor::dst_0);
    const auto dst_1 = TensorAccessor(tensor::dst_1);
    const auto dst_2 = TensorAccessor(tensor::dst_2);

    consume_one<implicit_sync>(dfb_0, dst_0, noc, num_entries, chunk_offset, entries_per_core);
    consume_one<implicit_sync>(dfb_1, dst_1, noc, num_entries, chunk_offset, entries_per_core);
    consume_one<implicit_sync>(dfb_2, dst_2, noc, num_entries, chunk_offset, entries_per_core);
}
