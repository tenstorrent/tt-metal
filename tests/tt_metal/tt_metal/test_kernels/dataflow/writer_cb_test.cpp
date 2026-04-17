// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#include "experimental/dataflow_buffer.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

template <bool use_dfbs, typename SyncBuffer>
inline __attribute__((always_inline)) void pop_from_cb_and_write(
    SyncBuffer& sync_buffer,
    uint32_t num_tiles_per_cb,
    uint32_t ublock_size_tiles,
    uint32_t ublock_size_bytes,
    uint32_t bank_id,
    uint32_t& dram_buffer_dst_addr) {
    experimental::Noc noc;
    for (uint32_t i = 0; i < num_tiles_per_cb; i += ublock_size_tiles) {
        sync_buffer.wait_front(ublock_size_tiles);
        noc.async_write(
            sync_buffer,
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dram_buffer_dst_addr});
        noc.async_write_barrier();
        sync_buffer.pop_front(ublock_size_tiles);
        dram_buffer_dst_addr += ublock_size_bytes;
    }
}

void kernel_main() {
    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t bank_id               = get_arg_val<uint32_t>(1);
    std::uint32_t num_tiles_per_cb      = get_arg_val<uint32_t>(2);

    constexpr uint32_t start_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_cbs_stride = get_compile_time_arg_val(1);
    constexpr uint32_t stride = get_compile_time_arg_val(2);
    constexpr uint32_t topmost_cb = get_compile_time_arg_val(3);
    constexpr uint32_t ublock_size_tiles = get_compile_time_arg_val(4);
    constexpr bool use_dfbs = get_compile_time_arg_val(5) == 1;
#ifdef ARCH_QUASAR
    static_assert(use_dfbs, "DFBs must be used on Quasar");
#endif

    // Process strided CBs: 0, 8, 16, ...
    for (uint32_t i = 0; i < num_cbs_stride; i++) {
        uint32_t cb_id = start_cb + i * stride;
#ifndef ARCH_QUASAR
        experimental::CircularBuffer cb(cb_id);
#endif
        experimental::DataflowBuffer dfb(cb_id);
        uint32_t ublock_size_bytes;
        if constexpr (use_dfbs) {
            ublock_size_bytes = dfb.get_entry_size() * ublock_size_tiles;
            pop_from_cb_and_write<true, experimental::DataflowBuffer>(
                dfb, num_tiles_per_cb, ublock_size_tiles, ublock_size_bytes, bank_id, dram_buffer_dst_addr);
        } else {
            ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;
            pop_from_cb_and_write<false, experimental::CircularBuffer>(
                cb, num_tiles_per_cb, ublock_size_tiles, ublock_size_bytes, bank_id, dram_buffer_dst_addr);
        }
    }
    // Process topmost CB
#ifndef ARCH_QUASAR
    experimental::CircularBuffer cb(topmost_cb);
#endif
    experimental::DataflowBuffer dfb(topmost_cb);
    uint32_t ublock_size_bytes;
    if constexpr (use_dfbs) {
        ublock_size_bytes = dfb.get_entry_size() * ublock_size_tiles;
        pop_from_cb_and_write<true, experimental::DataflowBuffer>(
            dfb, num_tiles_per_cb, ublock_size_tiles, ublock_size_bytes, bank_id, dram_buffer_dst_addr);
    } else {
        ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;
        pop_from_cb_and_write<false, experimental::CircularBuffer>(
            cb, num_tiles_per_cb, ublock_size_tiles, ublock_size_bytes, bank_id, dram_buffer_dst_addr);
    }
}
