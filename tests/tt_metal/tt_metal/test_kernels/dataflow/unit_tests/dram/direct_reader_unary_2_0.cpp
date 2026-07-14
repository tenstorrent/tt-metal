// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of direct_reader_unary.cpp. Streams DRAM tiles
// into a DataflowBuffer (DFB) declared on the host side; the legacy
// CircularBuffer path remains in direct_reader_unary.cpp for callers still on
// the legacy host API.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/kernel_thread_globals.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // DRAM page stride: the allocator may round page_size up (e.g. to
    // NOC_DRAM_READ_ALIGNMENT_BYTES = 64 on Quasar), so tiles are spaced
    // further apart in DRAM than their native size. Callers pass
    // Buffer::aligned_page_size() here; the kernel advances the DRAM
    // pointer by this stride while the DFB streams native-size tiles into L1.
    uint32_t src_addr = get_arg(args::src_addr);        // global base address
    uint32_t src_bank_id = get_arg(args::src_bank_id);  // data is in one bank
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t dram_page_stride = get_arg(args::dram_page_stride);

    constexpr uint32_t ublock_size_tiles = 1;

    const uint32_t producer_idx = get_my_thread_id();

    constexpr AllocatorBankType bank_type = AllocatorBankType::DRAM;
    AllocatorBank<bank_type> src_dram;
    Noc noc;

    DataflowBuffer dfb(dfb::out);
    uint32_t ublock_size_bytes = dfb.get_entry_size();
    // stride_factor = stride_in_entries: how many entry-sized slots one
    // producer skips per tile. For a DFB with N producers interleaved
    // round-robin, stride_factor = N, and each producer walks DRAM tiles
    // [producer_idx, producer_idx+N, producer_idx+2N, ...].
    uint32_t stride_factor = dfb.get_stride_size() / ublock_size_bytes;
    uint32_t tlocal_src_addr = src_addr + (producer_idx * dram_page_stride);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#ifdef ARCH_QUASAR
        // NocOptions::TXN_ID uses the DFB-integrated trid path on Quasar.
        noc.async_read<NocOptions::TXN_ID>(src_dram, dfb, {.bank_id = src_bank_id, .addr = tlocal_src_addr}, {});
#else
        dfb.reserve_back(ublock_size_tiles);
        noc.async_read(src_dram, dfb, ublock_size_bytes, {.bank_id = src_bank_id, .addr = tlocal_src_addr}, {});
        noc.async_read_barrier();
        dfb.push_back(ublock_size_tiles);
#endif
        tlocal_src_addr += dram_page_stride * stride_factor;
    }
    dfb.finish();
}
