// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Data moves of the fused attention glue kernels (data-movement RISCs only).
//
// Row form as rowio.hpp: a [.., 32, C] TILE tensor, row r of tile j holds elements 32j..32j+31 as two 16-element
// face-row chunks. A "head row" is 128 elements = 4 tiles = 8 chunks. Dense: elements contiguous in L1 (one page =
// 1024 elements, viewed as one 32x32 tile by compute).
#pragma once

#include <cstdint>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"

namespace attnio {

constexpr uint32_t ALIGN = 64;  // DRAM read alignment

FORCE_INLINE uint32_t row_offset(uint32_t r, uint32_t es) { return (r >> 4) * 512 * es + (r & 15) * 16 * es; }

// Zero `bytes` (multiple of 16) of L1 at dst with NOC reads of the hardware zero region.
FORCE_INLINE void zero_l1(uint32_t dst, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        const uint32_t n = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(z, dst, n);
        dst += n;
        bytes -= n;
    }
}

// Row r, tiles t0 .. t0 + ntc - 1 of a row-form DRAM tensor -> dense L1 at dst (32 * ntc elements), in two halves
// so that several rows share one read barrier: stage_row issues the 64B-aligned block reads into stage (ntc * 128
// bytes, 64B aligned), unstage_row (after a read barrier) the local chunk copies into dst.
// dram = false: the tensor is interleaved in L1 instead (same page layout).
template <bool dram = true>
FORCE_INLINE void stage_row(
    uint32_t base, uint32_t tile_bytes, uint32_t es, uint32_t t0, uint32_t ntc, uint32_t r, uint32_t stage) {
    const InterleavedAddrGen<dram> src = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t blk = row_offset(r, es) & ~(ALIGN - 1);
    for (uint32_t j = 0; j < ntc; ++j) {
        noc_async_read(src.get_noc_addr(t0 + j, blk), stage + j * 128, 64);
        noc_async_read(src.get_noc_addr(t0 + j, blk + 256 * es), stage + j * 128 + 64, 64);
    }
}

FORCE_INLINE void unstage_row(uint32_t es, uint32_t ntc, uint32_t r, uint32_t dst, uint32_t stage) {
    const uint32_t off = row_offset(r, es);
    const uint32_t in_blk = off - (off & ~(ALIGN - 1));
    const uint32_t chunk = 16 * es;
    for (uint32_t j = 0; j < ntc; ++j) {
        noc_async_read(get_noc_addr(stage + j * 128 + in_blk), dst + j * 2 * chunk, chunk);
        noc_async_read(get_noc_addr(stage + j * 128 + 64 + in_blk), dst + j * 2 * chunk + chunk, chunk);
    }
}

// stage_row + unstage_row with their barriers.
template <bool dram = true>
FORCE_INLINE void gather_row(
    uint32_t base,
    uint32_t tile_bytes,
    uint32_t es,
    uint32_t t0,
    uint32_t ntc,
    uint32_t r,
    uint32_t dst,
    uint32_t stage) {
    stage_row<dram>(base, tile_bytes, es, t0, ntc, r, stage);
    noc_async_read_barrier();
    unstage_row(es, ntc, r, dst, stage);
    noc_async_read_barrier();
}

// Issue the read of the 64B block holding element (r, c) of a row-form DRAM tensor (tile row starting at page t0)
// into stage; returns the L1 address the element will have there (valid after a read barrier).
template <bool dram = true>
FORCE_INLINE uint32_t stage_elem_block(
    uint32_t base, uint32_t tile_bytes, uint32_t es, uint32_t r, uint32_t c, uint32_t stage, uint32_t t0 = 0) {
    const InterleavedAddrGen<dram> src = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t cc = c & 31;
    const uint32_t off = row_offset(r, es) + (cc >> 4) * 256 * es + (cc & 15) * es;
    const uint32_t blk = off & ~(ALIGN - 1);
    noc_async_read(src.get_noc_addr(t0 + (c >> 5), blk), stage, 64);
    return stage + off - blk;
}

// The 64B block holding element (r, c) of a row-form DRAM tensor -> stage; returns the L1 address of the element.
FORCE_INLINE uint32_t
read_elem_block(uint32_t base, uint32_t tile_bytes, uint32_t es, uint32_t r, uint32_t c, uint32_t stage) {
    const InterleavedAddrGen<true> src = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t cc = c & 31;
    const uint32_t off = row_offset(r, es) + (cc >> 4) * 256 * es + (cc & 15) * es;
    const uint32_t blk = off & ~(ALIGN - 1);
    noc_async_read(src.get_noc_addr(c >> 5, blk), stage, 64);
    noc_async_read_barrier();
    return stage + off - blk;
}

// Dense head row (128 elements, es bytes each, at L1 src) -> row r of tiles t0 .. t0 + 3 of a TILE tensor.
// l1_noc != 0: the tensor is one L1 shard at (noc_x, noc_y) with tiles contiguous from base; else interleaved DRAM.
// Caller issues the write barrier.
FORCE_INLINE void scatter_head(
    uint32_t base,
    uint32_t tile_bytes,
    uint32_t es,
    uint32_t t0,
    uint32_t r,
    uint32_t src,
    uint32_t l1_noc,
    uint32_t noc_x,
    uint32_t noc_y) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t off = row_offset(r, es);
    const uint32_t chunk = 16 * es;
    for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t o = off + h * 256 * es;
            const uint64_t a =
                l1_noc ? get_noc_addr(noc_x, noc_y, base + (t0 + j) * tile_bytes + o) : dst.get_noc_addr(t0 + j, o);
            noc_async_write(src + (2 * j + h) * chunk, a, chunk);
        }
    }
}

// First 4 bytes of a DRAM buffer (an int32 / uint32 [1] tensor).
FORCE_INLINE uint32_t read_u32(uint32_t base, uint32_t page_bytes, uint32_t stage) {
    const InterleavedAddrGen<true> src = {.bank_base_address = base, .page_size = page_bytes};
    noc_async_read(src.get_noc_addr(0, 0), stage, 64);
    noc_async_read_barrier();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
}

FORCE_INLINE void read_pages(uint32_t base, uint32_t page_bytes, uint32_t first, uint32_t n, uint32_t dst) {
    const InterleavedAddrGen<true> src = {.bank_base_address = base, .page_size = page_bytes};
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(src.get_noc_addr(first + i), dst + i * page_bytes, page_bytes);
    }
}

}  // namespace attnio
