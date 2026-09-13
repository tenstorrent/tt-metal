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
        // Implicit sync: one call per tensor page. The DFB completes each share itself (waits for
        // room for the whole share, lands the entries `stride` apart, moves the bookmark after the
        // last one), so the kernel looks the same on every ring.
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
            const uint32_t page_id = chunk_offset + tile_id * num_producers + producer_idx;
            if (page_id >= chunk_offset + entries_per_core) {
                break;
            }
            noc.async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = page_id}, {});
        }
#endif
    } else {
        // Explicit sync: one op moves this hart's whole share (1 entry on a plain ring, its part of
        // each block when the consumers are BLOCKED); entry i of a strided share sits i * stride in.
#ifdef ARCH_QUASAR
        const uint32_t share = dfb.get_produce_share();
        const uint32_t stride_bytes = entry_size * dfb.get_produce_stride_tiles();
#else
        const uint32_t share = 1;
        const uint32_t stride_bytes = entry_size;
#endif
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
