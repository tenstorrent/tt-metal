// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) DFB producer.
// Parallel to ../dfb_producer.cpp (positional CTAs) — uses named CTAs/RTAs,
// dfb::out for the buffer binding, tensor::src_tensor for the input tensor, and
// get_my_thread_id() for per-thread striping.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

#if defined(ARCH_QUASAR) && defined(DFB_TEST_TEMPORARIES)
// Takes the DFB by value: the copy's reads must count toward the same DFB as the caller's.
template <typename Dfb, typename Accessor>
void read_through_copy(Dfb copy, const Noc& noc, const Accessor& accessor, uint32_t page_id) {
    noc.async_read<NocOptions::TXN_ID>(accessor, copy, {.page_id = page_id}, {});
}
#endif

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t chunk_offset = get_arg(args::chunk_offset);
    const uint32_t entries_per_core = get_arg(args::entries_per_core);

    DataflowBuffer dfb(dfb::out);
    Noc noc;
    const auto tensor_accessor = TensorAccessor(tensor::src_tensor);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    const uint32_t entry_size = dfb.get_entry_size();

    if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
        // Implicit sync: one call per page; the DFB does the reserve/push itself.
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
            const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
            if (page_id >= chunk_offset + entries_per_core) {
                break;
            }
#ifdef DFB_TEST_TEMPORARIES
            // Rotate through a temporary, a copy and the main object.
            if (tile_id % 3 == 0) {
                DataflowBuffer temporary(dfb::out);
                noc.async_read<NocOptions::TXN_ID>(tensor_accessor, temporary, {.page_id = page_id}, {});
            } else if (tile_id % 3 == 1) {
                read_through_copy(dfb, noc, tensor_accessor, page_id);
            } else
#endif
            {
                noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = page_id}, {});
            }
        }
#endif
    } else {
        // Explicit sync: one reserve/push covers share entries, spaced stride_bytes apart in the ring.
        const uint32_t share = dfb.get_producer_share();
        const uint32_t stride_bytes = entry_size * dfb.get_producer_stride_tiles();
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id += share) {
            const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
            if (page_id >= chunk_offset + entries_per_core) {
                break;
            }
            dfb.reserve_back(share);
            for (uint32_t i = 0; i < share; ++i) {
                noc.async_read(
                    tensor_accessor,
                    dfb,
                    entry_size,
                    {.page_id = page_id + i * num_producers},
                    {.offset_bytes = i * stride_bytes});
            }
            noc.async_read_barrier();
            dfb.push_back(share);
        }
    }
    dfb.finish();
}
