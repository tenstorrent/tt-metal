// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Row-form <-> dense moves for the Inkling-2-Mini fused residual kernels (data-movement RISCs only).
//
// Row form: a [.., R, C] TILE tensor in DRAM (interleaved); vector r lives in tile row r, so tile j holds its
// elements 32j..32j+31 as two 16-element face-row chunks (faces 0/1 for r < 16, faces 2/3 for r >= 16).
// Dense: the C elements of one vector contiguous in L1, padded with zeros up to a whole number of 1024-element
// pages. The compute kernels view each dense page as one 32x32 tile: fine for elementwise math and for full
// reductions, which is all they do with it.
#pragma once

#include <cstdint>

#include "api/alignment.h"
#include "api/dataflow/dataflow_api.h"

namespace rowio {

constexpr uint32_t DRAM_RD_ALIGN = 64;

// Byte offset of row r's first chunk inside a tile with es-byte elements (second chunk: + 256 * es).
FORCE_INLINE uint32_t row_offset(uint32_t r, uint32_t es) { return (r >> 4) * 512 * es + (r & 15) * 16 * es; }

FORCE_INLINE void zero_words(uint32_t l1, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

// Gather row r of a row-form tensor (base addr, tile_bytes per page, es bytes per element, ntc tiles along C)
// into dense L1 at dst (C = 32 * ntc elements; the pad up to pad_bytes is zeroed). stage: 64B-aligned scratch of
// ntc * 128 bytes. DRAM reads must be 64B aligned, so whole 64B blocks are staged and the chunks copied out.
// [j0, j1): only these tiles (the two data-movement RISCs can each gather a part; the pad goes with the last part).
template <bool DRAM = true>  // false: an L1-interleaved input (the MoE output)
FORCE_INLINE void gather_row(
    uint32_t base,
    uint32_t tile_bytes,
    uint32_t es,
    uint32_t ntc,
    uint32_t r,
    uint32_t dst,
    uint32_t pad_bytes,
    uint32_t stage,
    uint32_t j0 = 0,
    uint32_t j1 = 0xffffffff) {
    j1 = j1 < ntc ? j1 : ntc;
    const InterleavedAddrGen<DRAM> src = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t off = row_offset(r, es);
    const uint32_t blk = off & ~(DRAM_RD_ALIGN - 1);
    const uint32_t in_blk = off - blk;
    const uint32_t chunk = 16 * es;
    const uint32_t row_bytes = ntc * 32 * es;
    if (pad_bytes > row_bytes && j1 == ntc) {
        zero_words(dst + row_bytes, pad_bytes - row_bytes);
    }
    if (in_blk == 0 && chunk == 64 && (dst & (DRAM_RD_ALIGN - 1)) == 0) {  // fp32: chunks land in place
        for (uint32_t j = j0; j < j1; ++j) {
            noc_async_read(src.get_noc_addr(j, off), dst + j * 128, 64);
            noc_async_read(src.get_noc_addr(j, off + 1024), dst + j * 128 + 64, 64);
        }
        noc_async_read_barrier();
        return;
    }
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_read(src.get_noc_addr(j, blk), stage + j * 128, 64);
        noc_async_read(src.get_noc_addr(j, blk + 256 * es), stage + j * 128 + 64, 64);
    }
    noc_async_read_barrier();
    // chunks out of the staged blocks: local L1 -> L1 NOC copies (16B alignment), not RISC-V word loops
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_read(get_noc_addr(stage + j * 128 + in_blk), dst + j * 2 * chunk, chunk);
        noc_async_read(get_noc_addr(stage + j * 128 + 64 + in_blk), dst + j * 2 * chunk + chunk, chunk);
    }
    noc_async_read_barrier();
}

// Scatter dense L1 src (C = 32 * ntc elements, es bytes each) into row r of a row-form DRAM tensor. DRAM writes
// need 16B alignment only, which every chunk has. Caller issues the write barrier.
template <bool DRAM = true>  // false: an L1-interleaved output
// [j0, j1): only these tiles (both data-movement RISCs can each scatter a part).
FORCE_INLINE void scatter_row(
    uint32_t base,
    uint32_t tile_bytes,
    uint32_t es,
    uint32_t ntc,
    uint32_t r,
    uint32_t src,
    uint32_t j0 = 0,
    uint32_t j1 = 0xffffffff) {
    j1 = j1 < ntc ? j1 : ntc;
    const InterleavedAddrGen<DRAM> dst = {.bank_base_address = base, .page_size = tile_bytes};
    const uint32_t off = row_offset(r, es);
    const uint32_t chunk = 16 * es;
    for (uint32_t j = j0; j < j1; ++j) {
        noc_async_write(src + j * 2 * chunk, dst.get_noc_addr(j, off), chunk);
        noc_async_write(src + j * 2 * chunk + chunk, dst.get_noc_addr(j, off + 256 * es), chunk);
    }
}

// Whole dense pages (4 KiB fp32 / 2 KiB bf16) of an interleaved DRAM tensor <-> L1.
FORCE_INLINE void read_pages(uint32_t base, uint32_t page_bytes, uint32_t first, uint32_t n, uint32_t dst) {
    const InterleavedAddrGen<true> src = {.bank_base_address = base, .page_size = page_bytes};
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(src.get_noc_addr(first + i), dst + i * page_bytes, page_bytes);
    }
}

FORCE_INLINE void write_pages(uint32_t base, uint32_t page_bytes, uint32_t first, uint32_t n, uint32_t src) {
    const InterleavedAddrGen<true> dst = {.bank_base_address = base, .page_size = page_bytes};
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_write(src + i * page_bytes, dst.get_noc_addr(first + i), page_bytes);
    }
}

}  // namespace rowio
