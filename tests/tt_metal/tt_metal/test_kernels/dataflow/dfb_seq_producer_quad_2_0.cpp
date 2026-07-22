// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) sequential cooperative DM producer for 4-DFB
// TC-exhaustion tests. Parallel to ../dfb_seq_producer.cpp; M2 uses 4 distinct
// named DFB bindings (dfb::buf_0..buf_3) and 4 named tensor bindings
// (tensor::src_0..src_3) instead of the legacy runtime-determined DFB count.
//
// All threads cooperate on dfb::buf_0 first (each handling its own strided
// slice), then on dfb::buf_1, then buf_2, then buf_3. dfb.finish() at the end
// of each DFB acts as the cross-thread barrier.
//
// DataflowBuffer / TensorAccessor objects are constructed at function scope
// (matches the pattern in alias_dfb_producer.cpp).

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

template <typename Dfb, typename Acc>
static inline void produce_one_dfb(
    Dfb& dfb,
    const Acc& tensor_accessor,
    Noc& noc,
    uint32_t num_entries_per_producer,
    uint32_t num_producers,
    uint32_t producer_idx) {
    const uint32_t entry_size = dfb.get_entry_size();
    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        const uint32_t page_id = tile_id * num_producers + producer_idx;
        dfb.reserve_back(1);
        noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
    dfb.finish();
}

#ifdef ARCH_QUASAR
template <typename Dfb, typename Acc>
static inline void produce_one_dfb_impl_sync(
    Dfb& dfb,
    const Acc& tensor_accessor,
    Noc& noc,
    uint32_t num_entries_per_producer,
    uint32_t num_producers,
    uint32_t producer_idx) {
    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; ++tile_id) {
        const uint32_t page_id = tile_id * num_producers + producer_idx;
        noc.template async_read<NocOptions::TXN_ID>(tensor_accessor, dfb, {.page_id = page_id}, {});
    }
    dfb.finish();
}
#endif

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);

    const uint32_t producer_idx = get_my_thread_id();
    const uint32_t num_producers = get_num_threads();
    Noc noc;

    DataflowBuffer dfb_0(dfb::buf_0);
    DataflowBuffer dfb_1(dfb::buf_1);
    DataflowBuffer dfb_2(dfb::buf_2);
    DataflowBuffer dfb_3(dfb::buf_3);
    const auto src_0 = TensorAccessor(tensor::src_0);
    const auto src_1 = TensorAccessor(tensor::src_1);
    const auto src_2 = TensorAccessor(tensor::src_2);
    const auto src_3 = TensorAccessor(tensor::src_3);

    if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
        produce_one_dfb_impl_sync(dfb_0, src_0, noc, num_entries_per_producer, num_producers, producer_idx);
        produce_one_dfb_impl_sync(dfb_1, src_1, noc, num_entries_per_producer, num_producers, producer_idx);
        produce_one_dfb_impl_sync(dfb_2, src_2, noc, num_entries_per_producer, num_producers, producer_idx);
        produce_one_dfb_impl_sync(dfb_3, src_3, noc, num_entries_per_producer, num_producers, producer_idx);
#endif
    } else {
        produce_one_dfb(dfb_0, src_0, noc, num_entries_per_producer, num_producers, producer_idx);
        produce_one_dfb(dfb_1, src_1, noc, num_entries_per_producer, num_producers, producer_idx);
        produce_one_dfb(dfb_2, src_2, noc, num_entries_per_producer, num_producers, producer_idx);
        produce_one_dfb(dfb_3, src_3, noc, num_entries_per_producer, num_producers, producer_idx);
    }
}
