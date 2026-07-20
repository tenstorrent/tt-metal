// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) sequential cooperative DM consumer for 4-DFB
// TC-exhaustion tests. Parallel to ../dfb_seq_consumer.cpp.
//
// All threads cooperate on dfb::buf_0 first, then buf_1, buf_2, buf_3.
// Per-DFB access pattern (STRIDED vs ALL) is selected by compile-time args:
//   is_blocked_0..is_blocked_3
// When is_blocked is true, each consumer reads all entries (broadcast — ALL).
// When false, consumers stripe across pages (STRIDED).
//
// DataflowBuffer / TensorAccessor objects are constructed at function scope
// (matches the pattern in alias_dfb_consumer.cpp).

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

template <typename Dfb, typename Acc, bool is_blocked>
static inline void consume_one_dfb(
    Dfb& dfb,
    const Acc& tensor_accessor,
    Noc& noc,
    uint32_t entries_per_consumer,
    uint32_t num_consumers,
    uint32_t consumer_idx) {
    const uint32_t entry_size = dfb.get_entry_size();
    for (uint32_t tile_id = 0; tile_id < entries_per_consumer; ++tile_id) {
        const uint32_t page_id = is_blocked ? tile_id : tile_id * num_consumers + consumer_idx;
        dfb.wait_front(1);
        noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
        noc.async_write_barrier();
        dfb.pop_front(1);
    }
    dfb.finish();
    dfb.write_barrier(noc);
}

#ifdef ARCH_QUASAR
template <typename Dfb, typename Acc, bool is_blocked>
static inline void consume_one_dfb_impl_sync(
    Dfb& dfb,
    const Acc& tensor_accessor,
    Noc& noc,
    uint32_t entries_per_consumer,
    uint32_t num_consumers,
    uint32_t consumer_idx) {
    for (uint32_t tile_id = 0; tile_id < entries_per_consumer; ++tile_id) {
        const uint32_t page_id = is_blocked ? tile_id : tile_id * num_consumers + consumer_idx;
        noc.template async_write<NocOptions::TXN_ID>(dfb, tensor_accessor, {}, {.page_id = page_id});
    }
    dfb.finish();
    dfb.write_barrier(noc);
}
#endif

void kernel_main() {
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t entries_per_consumer_strided = get_arg(args::entries_per_consumer_strided);
    constexpr uint32_t entries_per_consumer_all = get_arg(args::entries_per_consumer_all);
    constexpr uint32_t is_blocked_0 = get_arg(args::is_blocked_0);
    constexpr uint32_t is_blocked_1 = get_arg(args::is_blocked_1);
    constexpr uint32_t is_blocked_2 = get_arg(args::is_blocked_2);
    constexpr uint32_t is_blocked_3 = get_arg(args::is_blocked_3);

    constexpr uint32_t epc_0 = is_blocked_0 ? entries_per_consumer_all : entries_per_consumer_strided;
    constexpr uint32_t epc_1 = is_blocked_1 ? entries_per_consumer_all : entries_per_consumer_strided;
    constexpr uint32_t epc_2 = is_blocked_2 ? entries_per_consumer_all : entries_per_consumer_strided;
    constexpr uint32_t epc_3 = is_blocked_3 ? entries_per_consumer_all : entries_per_consumer_strided;

    const uint32_t consumer_idx = get_my_thread_id();
    const uint32_t num_consumers = get_num_threads();
    Noc noc;

    DataflowBuffer dfb_0(dfb::buf_0);
    DataflowBuffer dfb_1(dfb::buf_1);
    DataflowBuffer dfb_2(dfb::buf_2);
    DataflowBuffer dfb_3(dfb::buf_3);
    const auto dst_0 = TensorAccessor(tensor::dst_0);
    const auto dst_1 = TensorAccessor(tensor::dst_1);
    const auto dst_2 = TensorAccessor(tensor::dst_2);
    const auto dst_3 = TensorAccessor(tensor::dst_3);

    if constexpr (implicit_sync) {
#ifdef ARCH_QUASAR
        consume_one_dfb_impl_sync<DataflowBuffer, decltype(dst_0), is_blocked_0>(
            dfb_0, dst_0, noc, epc_0, num_consumers, consumer_idx);
        consume_one_dfb_impl_sync<DataflowBuffer, decltype(dst_1), is_blocked_1>(
            dfb_1, dst_1, noc, epc_1, num_consumers, consumer_idx);
        consume_one_dfb_impl_sync<DataflowBuffer, decltype(dst_2), is_blocked_2>(
            dfb_2, dst_2, noc, epc_2, num_consumers, consumer_idx);
        consume_one_dfb_impl_sync<DataflowBuffer, decltype(dst_3), is_blocked_3>(
            dfb_3, dst_3, noc, epc_3, num_consumers, consumer_idx);
#endif
    } else {
        consume_one_dfb<DataflowBuffer, decltype(dst_0), is_blocked_0>(
            dfb_0, dst_0, noc, epc_0, num_consumers, consumer_idx);
        consume_one_dfb<DataflowBuffer, decltype(dst_1), is_blocked_1>(
            dfb_1, dst_1, noc, epc_1, num_consumers, consumer_idx);
        consume_one_dfb<DataflowBuffer, decltype(dst_2), is_blocked_2>(
            dfb_2, dst_2, noc, epc_2, num_consumers, consumer_idx);
        consume_one_dfb<DataflowBuffer, decltype(dst_3), is_blocked_3>(
            dfb_3, dst_3, noc, epc_3, num_consumers, consumer_idx);
    }
}
