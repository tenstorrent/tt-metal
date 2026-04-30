// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/bound_buffer.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#ifndef ARCH_QUASAR
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr bool use_dfbs = get_compile_time_arg_val(1) == 1;

    // Buffer descriptor (base address, bank_id, num_tiles, per-tile DRAM
    // stride) is supplied at runtime-arg slot 0..3 by BindBufferToKernel on
    // the host. The previously-explicit `dram_page_stride` arg is now hidden
    // inside BoundBuffer — kernel devs no longer see DRAM page-alignment vs.
    // native tile size asymmetry.
    experimental::BoundBuffer<experimental::AllocatorBankType::DRAM> src(/*arg_slot=*/0);
    const uint32_t num_tiles = src.num_tiles();

    constexpr uint32_t ublock_size_tiles = 1;

#ifdef ARCH_QUASAR
    uint32_t producer_idx = get_my_thread_id();
#else
    uint32_t producer_idx = 0;
#endif

    experimental::Noc noc;
    if constexpr (use_dfbs) {
        experimental::DataflowBuffer dfb(cb_id);
        uint32_t ublock_size_bytes = dfb.get_entry_size();
        // stride_factor = stride_in_entries: how many entry-sized slots one
        // producer skips per tile. For a DFB with N producers interleaved
        // round-robin, stride_factor = N, and each producer walks DRAM tiles
        // [producer_idx, producer_idx+N, producer_idx+2N, ...].
        uint32_t stride_factor = dfb.get_stride_size() / ublock_size_bytes;

        for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
            const uint32_t addr = src.tile_addr(producer_idx + i * stride_factor);
#ifdef ARCH_QUASAR
            noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
                src.bank(), dfb, {.bank_id = src.bank_id(), .addr = addr}, {});
#else
            dfb.reserve_back(ublock_size_tiles);
            noc.async_read(src.bank(), dfb, ublock_size_bytes, {.bank_id = src.bank_id(), .addr = addr}, {});
            noc.async_read_barrier();
            dfb.push_back(ublock_size_tiles);
#endif
        }
        dfb.finish();
    }
#ifndef ARCH_QUASAR
    else {
        experimental::CircularBuffer cb(cb_id);
        uint32_t ublock_size_bytes = cb.get_tile_size() * ublock_size_tiles;

        for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
            cb.reserve_back(ublock_size_tiles);
            noc.async_read(src.bank(), cb, ublock_size_bytes, {.bank_id = src.bank_id(), .addr = src.tile_addr(i)}, {});
            noc.async_read_barrier();
            cb.push_back(ublock_size_tiles);
        }
    }
#endif
}
