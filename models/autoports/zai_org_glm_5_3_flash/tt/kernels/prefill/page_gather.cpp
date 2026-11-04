// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 P3 (tt/prefill_tp.py, DSA sparse prefill): gather a slot's pages of a paged TILE cache into a contiguous
// TILE tensor, the page list read from a device tensor (the slot's page-table row), so one captured program serves
// every slot. A page = TPP consecutive tiles of the source ([pages, 1, rows, W] TILE: page p = tiles [p TPP, (p+1)
// TPP)), destination page i = tiles [i TPP, (i+1) TPP) of the contiguous [1, 1, n_pages rows, W] tensor.
//
// Compile-time: TPP (tiles per page), TILE_BYTES, TABLE_BYTES (the int32 table row, one ROW_MAJOR page), DEPTH (tiles
// in flight). Runtime per core: src, dst, table addresses, first destination page, page count of this core.
// CB GATHER_CB: TABLE_BYTES (aligned) + DEPTH x TILE_BYTES (+ 64 B for the bound word under PAGE_BOUND).
// PAGE_BOUND (GLM53_PF_IX_GATHER_BOUND, the indexer's visible-pool bound stage 2): runtime arg 5 = a one-element uint32
// device tensor (DRAM, interleaved: the per-chip word host-written per chunk, so a captured program serves every chunk
// start), compile arg 5 = BOUND_DIV: this core handles only destination pages i < word / BOUND_DIV of its range.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPP = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t TABLE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t TABLE_ALIGNED = get_compile_time_arg_val(3);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(4);
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t first = get_arg_val<uint32_t>(3);
    uint32_t count = get_arg_val<uint32_t>(4);
    if (count == 0) {
        return;
    }

    uint32_t base = get_write_ptr(GATHER_CB);
#ifdef PAGE_BOUND
    {
        constexpr uint32_t BOUND_DIV = get_compile_time_arg_val(5);
        const uint32_t bound_addr = get_arg_val<uint32_t>(5);
        // the word in its OWN 64-byte slot (a Blackhole DRAM read needs 64-byte alignment; expand_rows 2026-10-21)
        const InterleavedAddrGen<true> bound_g{.bank_base_address = bound_addr, .page_size = 4};
        noc_async_read(bound_g.get_noc_addr(0), base, 64);
        noc_async_read_barrier();
        const uint32_t lim = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[0] / BOUND_DIV;
        base += 64;
        count = lim <= first ? 0u : (first + count <= lim ? count : lim - first);
        if (count == 0) {
            return;
        }
    }
#endif
    const InterleavedAddrGen<true> table_g{.bank_base_address = table_addr, .page_size = TABLE_BYTES};
    noc_async_read(table_g.get_noc_addr(0), base, TABLE_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base);
    const uint32_t buf = base + TABLE_ALIGNED;
    const InterleavedAddrGen<true> src{.bank_base_address = src_addr, .page_size = TILE_BYTES};
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = TILE_BYTES};

    uint32_t slot = 0;
    for (uint32_t i = first; i < first + count; ++i) {
        const int32_t p = table[i];
#ifdef PAGE_SCATTER  // the reverse: dst page table[i] <- src page i
        const uint32_t s0 = i * TPP;
        const uint32_t d0 = (p < 0 ? 0u : static_cast<uint32_t>(p)) * TPP;
#else
        const uint32_t s0 = (p < 0 ? 0u : static_cast<uint32_t>(p)) * TPP;
        const uint32_t d0 = i * TPP;
#endif
        for (uint32_t j = 0; j < TPP; ++j) {
            noc_async_read(src.get_noc_addr(s0 + j), buf + slot * TILE_BYTES, TILE_BYTES);
            if (++slot == DEPTH || j + 1 == TPP) {
                noc_async_read_barrier();
                // write back the tiles just read (destination tiles d0 + j - slot + 1 .. d0 + j)
                for (uint32_t k = 0; k < slot; ++k) {
                    noc_async_write(buf + k * TILE_BYTES, dst.get_noc_addr(d0 + j + 1 - slot + k), TILE_BYTES);
                }
                noc_async_write_barrier();
                slot = 0;
            }
        }
    }
}
