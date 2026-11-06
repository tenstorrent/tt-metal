// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// region_plan data-movement kernel (model-local generic op: custom_ops/region_plan/op.py), one core. The regions
// prefill path's bookkeeping in one program: from the router's top-k expert ids (uint32 [T*k], row-major (t, j)) and
// the chip's local expert list (gidx [E_loc] global ids), produce
//   counts[e]  = the chip's picks of global expert e (0 off-chip)              uint32 [E]
//   offsets[e] = tile-aligned exclusive prefix of counts over e               uint32 [E]
//   pos[t*k+j] = offsets[e] + rank of the pick among e's picks (local), else max_buf - 1   uint32 [T*k]
//   loc        = 1.0 at the local picks, 0.0 elsewhere, as the fp32 TILE tensor [T, k] (k <= 16: faces 0 / 2)
// Two passes over the ids in 64 KB chunks (counts, then ranks + the pos chunks + the loc tiles).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t E_loc = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_entries = get_compile_time_arg_val(4);  // a multiple of 32 * K (whole tile rows)
    constexpr uint32_t tile_bytes = 4096u;                           // fp32 32 x 32
    constexpr uint32_t TILE = 32u;
    constexpr auto ids_a = TensorAccessorArgs<5>();
    constexpr auto gidx_a = TensorAccessorArgs<ids_a.next_compile_time_args_offset()>();
    constexpr auto counts_a = TensorAccessorArgs<gidx_a.next_compile_time_args_offset()>();
    constexpr auto offsets_a = TensorAccessorArgs<counts_a.next_compile_time_args_offset()>();
    constexpr auto pos_a = TensorAccessorArgs<offsets_a.next_compile_time_args_offset()>();
    constexpr auto loc_a = TensorAccessorArgs<pos_a.next_compile_time_args_offset()>();

    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t gidx_addr = get_arg_val<uint32_t>(1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(2);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t loc_addr = get_arg_val<uint32_t>(5);
    const uint32_t max_buf = get_arg_val<uint32_t>(6);
    const uint32_t ids_page_bytes = get_arg_val<uint32_t>(7);
    const uint32_t gidx_page_bytes = get_arg_val<uint32_t>(8);
    const uint32_t e_page_bytes = get_arg_val<uint32_t>(9);  // counts / offsets: one page of E entries (64 B aligned)

    const uint32_t n_entries = T * K;
    const auto ids_acc = TensorAccessor(ids_a, ids_addr, ids_page_bytes);
    const auto gidx_acc = TensorAccessor(gidx_a, gidx_addr, gidx_page_bytes);
    const auto counts_acc = TensorAccessor(counts_a, counts_addr, e_page_bytes);
    const auto offsets_acc = TensorAccessor(offsets_a, offsets_addr, e_page_bytes);
    const auto pos_acc = TensorAccessor(pos_a, pos_addr, ids_page_bytes);
    const auto loc_acc = TensorAccessor(loc_a, loc_addr, tile_bytes);

    auto align64 = [](uint32_t a) { return (a + 63u) & ~63u; };
    const uint32_t ids_base = align64(CircularBuffer(0).get_write_ptr());   // chunk_entries x 4 B
    const uint32_t pos_base = align64(CircularBuffer(1).get_write_ptr());   // chunk_entries x 4 B
    const uint32_t tab_base = align64(CircularBuffer(2).get_write_ptr());   // E x 4 (local map) + E x 4 (counts)
    const uint32_t off_base = align64(tab_base + E * 4u);                   // offsets E x 4
    const uint32_t rank_base = align64(off_base + E * 4u);                  // ranks E x 4
    const uint32_t cnt_base = align64(rank_base + E * 4u);                  // counts E x 4
    const uint32_t gidx_base = align64(cnt_base + E * 4u);                  // gidx E_loc x 4
    const uint32_t tile_base = align64(CircularBuffer(3).get_write_ptr());  // one fp32 tile

    Noc noc;
    // the local map: global expert -> local index, 0xFFFFFFFF off-chip
    noc.async_read(gidx_acc, CoreLocalMem<uint32_t>(gidx_base), align64(E_loc * 4u), {.page_id = 0}, {});
    noc.async_read_barrier();
    CoreLocalMem<volatile uint32_t> tab(tab_base);
    CoreLocalMem<volatile uint32_t> cnt(cnt_base);
    CoreLocalMem<volatile uint32_t> off(off_base);
    CoreLocalMem<volatile uint32_t> rank(rank_base);
    const CoreLocalMem<volatile uint32_t> gidx(gidx_base);
    for (uint32_t e = 0; e < E; e++) {
        tab[e] = 0xFFFFFFFFu;
        cnt[e] = 0u;
        rank[e] = 0u;
    }
    for (uint32_t i = 0; i < E_loc; i++) {
        const uint32_t g = gidx[i];
        if (g < E) {
            tab[g] = i;
        }
    }

    // pass 1: counts
    const CoreLocalMem<volatile uint32_t> ids(ids_base);
    for (uint32_t base = 0; base < n_entries; base += chunk_entries) {
        const uint32_t n = (n_entries - base < chunk_entries) ? (n_entries - base) : chunk_entries;
        noc.async_read(
            ids_acc, CoreLocalMem<uint32_t>(ids_base), align64(n * 4u), {.page_id = 0, .offset_bytes = base * 4u}, {});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t e = ids[i];
            if (e < E && tab[e] != 0xFFFFFFFFu) {
                cnt[e] = cnt[e] + 1u;
            }
        }
    }
    // tile-aligned exclusive prefix over e
    {
        uint32_t acc = 0u;
        for (uint32_t e = 0; e < E; e++) {
            off[e] = acc;
            const uint32_t c = cnt[e];
            acc += ((c + TILE - 1u) / TILE) * TILE;
        }
    }
    noc.async_write(CoreLocalMem<uint32_t>(cnt_base), counts_acc, align64(E * 4u), {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint32_t>(off_base), offsets_acc, align64(E * 4u), {}, {.page_id = 0});

    // pass 2: ranks -> pos chunks, loc tiles (K columns of a 32-wide fp32 tile: faces 0 (rows 0-15) and 2 (rows 16-31))
    CoreLocalMem<volatile uint32_t> pos(pos_base);
    CoreLocalMem<volatile uint32_t> tile(tile_base);
    const uint32_t one_f = 0x3F800000u;
    uint32_t tile_id = 0u;
    for (uint32_t base = 0; base < n_entries; base += chunk_entries) {
        const uint32_t n = (n_entries - base < chunk_entries) ? (n_entries - base) : chunk_entries;
        noc.async_write_barrier();  // the previous chunk's pos / tiles left the L1 buffers
        noc.async_read(
            ids_acc, CoreLocalMem<uint32_t>(ids_base), align64(n * 4u), {.page_id = 0, .offset_bytes = base * 4u}, {});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t e = ids[i];
            if (e < E && tab[e] != 0xFFFFFFFFu) {
                const uint32_t r = rank[e];
                rank[e] = r + 1u;
                pos[i] = off[e] + r;
            } else {
                pos[i] = max_buf - 1u;
            }
        }
        // the chunk's pos entries (8 KB pieces)
        for (uint32_t b = 0; b < n * 4u; b += 8192u) {
            const uint32_t len = (n * 4u - b < 8192u) ? (n * 4u - b) : 8192u;
            noc.async_write(
                CoreLocalMem<uint32_t>(pos_base + b),
                pos_acc,
                align64(len),
                {},
                {.page_id = 0, .offset_bytes = base * 4u + b});
        }
        // the loc tiles of this chunk: rows t = (base + i) / K
        const uint32_t rows = n / K;
        for (uint32_t r0 = 0; r0 < rows; r0 += TILE) {
            for (uint32_t w = 0; w < tile_bytes / 4u; w++) {
                tile[w] = 0u;
            }
            for (uint32_t rr = 0; rr < TILE && r0 + rr < rows; rr++) {
                for (uint32_t j = 0; j < K; j++) {
                    const uint32_t i = (r0 + rr) * K + j;
                    if (pos[i] != max_buf - 1u) {
                        const uint32_t face = (rr >= 16u) ? 2u : 0u;  // K <= 16: column face 0
                        const uint32_t idx = face * 256u + (rr & 15u) * 16u + j;
                        tile[idx] = one_f;
                    }
                }
            }
            noc.async_write(CoreLocalMem<uint32_t>(tile_base), loc_acc, tile_bytes, {}, {.page_id = tile_id});
            noc.async_write_barrier();  // one tile buffer: drain before the next compose
            tile_id++;
        }
    }
    noc.async_write_barrier();
}
