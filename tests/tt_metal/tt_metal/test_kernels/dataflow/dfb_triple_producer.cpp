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
static inline void produce_one(
    Dfb& dfb, const Acc& src, Noc& noc, uint32_t num_entries, uint32_t chunk_offset, uint32_t entries_per_core) {
    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();
    for (uint32_t tile = 0; tile < num_entries; ++tile) {
        const uint32_t page_id = chunk_offset + tile * num_producers + producer_idx;
        if (page_id >= chunk_offset + entries_per_core) {
            break;
        }
        if constexpr (ImplicitSync) {
#ifdef ARCH_QUASAR
            noc.async_read<NocOptions::TXN_ID>(src, dfb, {.page_id = page_id}, {});
#endif
        } else {
            dfb.reserve_back(1);
            noc.async_read(src, dfb, entry_size, {.page_id = page_id}, {});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    }
    dfb.finish();
}

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries_per_producer);
    constexpr bool implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    Noc noc;
    DataflowBuffer dfb_0(dfb::out_0);
    DataflowBuffer dfb_1(dfb::out_1);
    DataflowBuffer dfb_2(dfb::out_2);
    const auto src_0 = TensorAccessor(tensor::src_0);
    const auto src_1 = TensorAccessor(tensor::src_1);
    const auto src_2 = TensorAccessor(tensor::src_2);

    produce_one<implicit_sync>(dfb_0, src_0, noc, num_entries, chunk_offset, entries_per_core);
    produce_one<implicit_sync>(dfb_1, src_1, noc, num_entries, chunk_offset, entries_per_core);
    produce_one<implicit_sync>(dfb_2, src_2, noc, num_entries, chunk_offset, entries_per_core);
}
