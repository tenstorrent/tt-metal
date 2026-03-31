// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t dst_addr_base = get_compile_time_arg_val(0);
    const uint32_t blocked_consumer = get_compile_time_arg_val(1);
    constexpr uint32_t implicit_sync = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    uint32_t num_entries = get_arg_val<uint32_t>(0);
    uint32_t consumer_mask = get_arg_val<uint32_t>(1);
    uint32_t logical_dfb_id = get_arg_val<uint32_t>(2);
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    experimental::DataflowBuffer dfb(logical_dfb_id);
    experimental::Noc noc;

    // TODO: Replace with get_thread_idx() kernel API when available
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint32_t consumer_idx = static_cast<uint32_t>(__builtin_popcount(consumer_mask & ((1u << hartid) - 1u)));

    // DPRINT << "consumer_idx: " << consumer_idx << " num_entries: " << num_entries << ENDL();

    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base, entry_size);

    for (uint32_t tile_id = 0; tile_id < num_entries; tile_id++) {
        // Blocked: every consumer processes all tiles.
        // Strided: each consumer owns every num_consumers-th tile starting at consumer_idx.
        if constexpr (!blocked_consumer) {
            if (tile_id % num_consumers != consumer_idx) {
                continue;
            }
        }
        // tile_id is already the global page index into the output buffer
        if constexpr (implicit_sync) {
            dfb.write_out(noc, tensor_accessor, {.page_id = tile_id});
        } else {
            dfb.wait_front(1);
            noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = tile_id});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    }
    DPRINT << "CBW" << ENDL();
    dfb.finish();
    if constexpr (implicit_sync) {
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id];
        for (uint32_t i = 0; i < local_dfb_interface.num_txn_ids; i++) {
            noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>(local_dfb_interface.txn_ids[i]);
        }
    } else {
        noc.async_write_barrier();
    }
    DPRINT << "CBWD" << ENDL();
}
