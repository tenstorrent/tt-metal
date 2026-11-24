// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 indexer visible-pool bound, stage 2 (GLM53_PF_IX_GATHER_BOUND, tt/prefill_tp.py SparseCtx.paged_pools): this
// chip's pooled keys of the slot, COMPACT, bounded. Under GLM53_DSA_PAGED_POOLS a pooled-key page is the latent block
// page (a 64-token block: 32 rows x 128 = TPP tiles; its 16 pooled keys in rows 0-15 = faces 0-1 of every tile = the
// first HALF_BYTES of each tile). A shard unit = 128 tokens = local blocks 2 lu, 2 lu + 1 = 32 pools = ONE pool tile
// row, so compact tile row lu, tile j = [block 2 lu tile j faces 0-1 | block 2 lu + 1 tile j faces 0-1] (rows 0-15 |
// 16-31). The pages come from the slot's LOCAL table (int32 row, local block -> physical page of the store).
//
// Compile-time: TPP, TILE_BYTES, TABLE_BYTES, TABLE_ALIGNED. Runtime per core: src (the store), dst (the compact
// [n_units x 32, 128] TILE buffer), table, bound (one-element uint32 device word = this chip's local units to compact),
// first unit, unit count. CB GATHER_CB: 64 B (bound word) + TABLE_ALIGNED + 2 x TILE_BYTES.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TPP = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t TABLE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t TABLE_ALIGNED = get_compile_time_arg_val(3);
    constexpr uint32_t HALF_BYTES = TILE_BYTES / 2;  // faces 0-1 of a tile = its rows 0-15
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t bound_addr = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4);
    uint32_t count = get_arg_val<uint32_t>(5);
    if (count == 0) {
        return;
    }

    const uint32_t base = get_write_ptr(GATHER_CB);
    const InterleavedAddrGen<true> bound_g{.bank_base_address = bound_addr, .page_size = 4};
    noc_async_read(bound_g.get_noc_addr(0), base, 64);
    noc_async_read_barrier();
    const uint32_t lim = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[0];
    count = lim <= first ? 0u : (first + count <= lim ? count : lim - first);
    if (count == 0) {
        return;
    }
    const uint32_t tbl = base + 64;
    const InterleavedAddrGen<true> table_g{.bank_base_address = table_addr, .page_size = TABLE_BYTES};
    noc_async_read(table_g.get_noc_addr(0), tbl, TABLE_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(tbl);
    const uint32_t buf = tbl + TABLE_ALIGNED;  // two tiles: double-buffered output tiles
    const InterleavedAddrGen<true> src{.bank_base_address = src_addr, .page_size = TILE_BYTES};
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = TILE_BYTES};

    uint32_t slot = 0;
    for (uint32_t lu = first; lu < first + count; ++lu) {
        const int32_t pa_ = table[2 * lu];
        const int32_t pb_ = table[2 * lu + 1];
        const uint32_t pa = (pa_ < 0 ? 0u : static_cast<uint32_t>(pa_)) * TPP;
        const uint32_t pb = (pb_ < 0 ? 0u : static_cast<uint32_t>(pb_)) * TPP;
        for (uint32_t j = 0; j < TPP; ++j) {
            const uint32_t t = buf + slot * TILE_BYTES;
            if (slot == 0) {
                noc_async_write_barrier();  // this slot's previous tile left (two tiles in flight at most)
            }
            noc_async_read(src.get_noc_addr(pa + j), t, HALF_BYTES);
            noc_async_read(src.get_noc_addr(pb + j), t + HALF_BYTES, HALF_BYTES);
            noc_async_read_barrier();
            noc_async_write(t, dst.get_noc_addr(lu * TPP + j), TILE_BYTES);
            slot ^= 1;
        }
    }
    noc_async_write_barrier();
}
