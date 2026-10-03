// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Shared helpers of the attention "pre" program (heads split + RoPE + KV cache write). bf16 TILE tensors, DRAM
// interleaved.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

typedef volatile tt_l1_ptr uint32_t* PW32;

// scratch CB layout (bytes)
constexpr uint32_t SRC_OFF = 0;       // 9 source rows x 128 B (2 aligned 64 B chunks each)
constexpr uint32_t LAT_OFF = 1152;    // 128 B: latent row chunks
constexpr uint32_t IDX_OFF = 1280;    // 64 B x 2: position / compressed slot indices
constexpr uint32_t PATCH_OFF = 2048;  // 2 KB cache tile

// Row r of a bf16 tile, columns 0..15 / 16..31, 64 B aligned chunk offsets (holds rows r&~1 and r|1)
inline uint32_t chunk_l(uint32_t r) { return ((r >> 4) << 10) + (((r & 15) >> 1) << 6); }

// out tile (L1 words at `o`, rows 0..8) <- kvn row t (row 0) and q rows t of 8 head tiles (rows 1..8) of tile column n
template <typename K, typename Q>
inline void build_tile(
    Noc& noc, const K& kvn_acc, const Q& q_acc, experimental::CB& cbs, PW32 o, uint32_t t, uint32_t n) {
    const uint32_t cl = chunk_l(t);
    for (uint32_t k = 0; k < 9; ++k) {
        const uint32_t page = (k == 0) ? n : ((k - 1) * 16 + n);
        if (k == 0) {
            noc.async_read(kvn_acc, cbs, 64, {.page_id = page, .offset_bytes = cl}, {.offset_bytes = SRC_OFF});
            noc.async_read(
                kvn_acc, cbs, 64, {.page_id = page, .offset_bytes = cl + 512}, {.offset_bytes = SRC_OFF + 64});
        } else {
            noc.async_read(q_acc, cbs, 64, {.page_id = page, .offset_bytes = cl}, {.offset_bytes = SRC_OFF + k * 128});
            noc.async_read(
                q_acc, cbs, 64, {.page_id = page, .offset_bytes = cl + 512}, {.offset_bytes = SRC_OFF + k * 128 + 64});
        }
    }
    noc.async_read_barrier();
    PW32 s = reinterpret_cast<PW32>(cbs.get_write_ptr() + SRC_OFF);
    const uint32_t p = (t & 1) * 8;  // word offset of row parity inside a 64 B chunk
    for (uint32_t k = 0; k < 9; ++k) {
        for (uint32_t e = 0; e < 8; ++e) {
            o[k * 8 + e] = s[k * 32 + p + e];
            o[128 + k * 8 + e] = s[k * 32 + 16 + p + e];
        }
    }
}

template <typename C>
inline void patch_row(
    Noc& noc, const C& cache_acc, experimental::CB& cbs, uint32_t page, uint32_t row, PW32 left, PW32 right) {
    noc.async_read(cache_acc, cbs, 2048, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = PATCH_OFF});
    noc.async_read_barrier();
    PW32 p = reinterpret_cast<PW32>(cbs.get_write_ptr() + PATCH_OFF);
    const uint32_t w = ((row >> 4) << 8) + ((row & 15) << 3);
    for (uint32_t e = 0; e < 8; ++e) {
        p[w + e] = left[e];
        p[128 + w + e] = right[e];
    }
    noc.async_write(cbs, cache_acc, 2048, {.offset_bytes = PATCH_OFF}, {.page_id = page, .offset_bytes = 0});
    noc.async_write_barrier();
}

// KV (row 0 of the finished tile `o`) -> cache slot pos[t]; latent row t of tile column n -> cache slot comp[t]
// (HAS_LAT)
template <uint32_t HAS_LAT, uint32_t CACHE_PT, typename C, typename L, typename P, typename M>
inline void cache_writes(
    Noc& noc,
    const C& cache_acc,
    const L& lat_acc,
    const P& pos_acc,
    const M& comp_acc,
    experimental::CB& cbs,
    PW32 o,
    uint32_t t,
    uint32_t n) {
    noc.async_read(pos_acc, cbs, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = IDX_OFF});
    if (HAS_LAT) {
        noc.async_read(comp_acc, cbs, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = IDX_OFF + 64});
        const uint32_t cl = chunk_l(t);
        noc.async_read(lat_acc, cbs, 64, {.page_id = n, .offset_bytes = cl}, {.offset_bytes = LAT_OFF});
        noc.async_read(lat_acc, cbs, 64, {.page_id = n, .offset_bytes = cl + 512}, {.offset_bytes = LAT_OFF + 64});
    }
    noc.async_read_barrier();
    PW32 ix = reinterpret_cast<PW32>(cbs.get_write_ptr() + IDX_OFF);
    const uint32_t idx = ix[t];
    patch_row(noc, cache_acc, cbs, t * CACHE_PT + (idx >> 5) * 16 + n, idx & 31, o, o + 128);
    if (HAS_LAT) {
        const uint32_t cidx = ix[16 + t];
        PW32 l = reinterpret_cast<PW32>(cbs.get_write_ptr() + LAT_OFF);
        const uint32_t p = (t & 1) * 8;
        patch_row(noc, cache_acc, cbs, t * CACHE_PT + (cidx >> 5) * 16 + n, cidx & 31, l + p, l + 16 + p);
    }
}
