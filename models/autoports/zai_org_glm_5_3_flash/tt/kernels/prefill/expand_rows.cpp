// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 P3 (tt/prefill_tp.py expand_rows, the sparse_sdpa DSA prefill): DSA id expansion for every query row of a
// prefill chunk (prework/dsa/kernels/expand_selection.cpp, one decode token per user, applied per chunk row). Row r of
// this core's range [first, first + count) is the token at position p = start + base + r (start: a one-element int32
// device tensor, the chunk start; base: a one-element int32 tensor, this chip's first row in the chunk -- 0 when every
// chip expands all rows, rank x C_local when each chip expands its own sequence shard). With n_closed = (p + 1) /
// KPOOL complete pools, nsel = min(n_closed, TOPK) and A = KPOOL * nsel, output slot i of the row is
//   i <  A : KPOOL * pools[r][i / KPOOL] + i % KPOOL          (the selected pools, best first)
//   i >= A : t = KPOOL * n_closed + (i - A), valid while t <= p (the open pool: always attended), then 0xFFFFFFFF
// so the valid ids form a prefix and the sentinels a contiguous tail (the sparse_sdpa contract). Ids are token
// positions = rows of the request's ROW_MAJOR latent store. A pool >= n_closed among the first nsel (masked scores rank
// last, so it cannot happen with >= nsel visible pools) is replaced by the row's own position (a causal key).
//
// Compile-time: TOPK, KPOOL, WIDTH (ids per row), SHARD (0 / 1). Runtime per core: pools, start, out, base addresses,
// first row, row count [, SHARD: the shard page address, the counts address].
// pools [.., C, TOPK] uint32 ROW_MAJOR (one page per row), out [.., C, WIDTH] uint32 ROW_MAJOR.
// SHARD = 1 (the sequence-sharded DSA pool, tt/dsa_shard.py; lever B-q split-K prefill): this chip (rank r of C,
// read from the per-chip shard page [r, C, ..]) emits only the keys it OWNS, as LOCAL row ids of its latent slice
// (unit u = 128 tokens on chip u % C at local unit u // C: id' = ((g / 32 / C) 32 + g % 32) KPOOL + k for pool g,
// local_pos(t) for the tail), in the same order, then the sentinel tail; counts[r] word 0 = the keys emitted. A row
// this chip owns no key of gets ONE dummy key (local row 0: a written or null page, finite) so the op's "every row
// has >= 1 valid key" precondition holds; the merge zeroes that partial from the count (tt/prefill_tp.py).
// CB 0: 64 B (start) + 64 B (base) + 64 B (shard) + TOPK x 4 (aligned 64) + WIDTH x 4 + 64 B (counts). Each 4-byte
// scalar lands in its OWN 64-byte slot:
// a Blackhole DRAM read needs NOC_DRAM_READ_ALIGNMENT_BYTES (64) alignment, and the base read used to target base + 32,
// so it landed elsewhere and the kernel read a stale L1 word as the row base -- p off by that garbage, tail ids past
// the row's position (huge: out-of-range sparse_sdpa keys, ~1e36 rows) for 3 of 4 phases (2026-10-21 08:01Z).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t TOPK = get_compile_time_arg_val(0);
    constexpr uint32_t KPOOL = get_compile_time_arg_val(1);
    constexpr uint32_t WIDTH = get_compile_time_arg_val(2);
    constexpr bool SHARD = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t UNIT = 128, POOL_PAGE = 32;  // tt/dsa_shard.py UNIT / POOL_PAGE (unit = one pooled-key page)
    constexpr uint32_t POOLS_BYTES = TOPK * 4;
    constexpr uint32_t POOLS_ALIGNED = (POOLS_BYTES + 63) / 64 * 64;
    constexpr uint32_t OUT_BYTES = WIDTH * 4;
    constexpr uint32_t COUNTS_BYTES = 64;  // one 64-B page per row: word 0 = the keys this chip emitted
    const uint32_t pools_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t base_addr = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    if (count == 0) {
        return;
    }

    const uint32_t base = get_write_ptr(0);
    const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = 4};
    const InterleavedAddrGen<true> rbase_g{.bank_base_address = base_addr, .page_size = 4};
    noc_async_read(start_g.get_noc_addr(0), base, 4);
    noc_async_read(rbase_g.get_noc_addr(0), base + 64, 4);
    uint32_t rank = 0, nchips = 1;
    uint32_t counts_addr = 0;
    if constexpr (SHARD) {
        const uint32_t shard_addr = get_arg_val<uint32_t>(6);
        counts_addr = get_arg_val<uint32_t>(7);
        const InterleavedAddrGen<true> shard_g{.bank_base_address = shard_addr, .page_size = 64};
        noc_async_read(shard_g.get_noc_addr(0), base + 128, 64);
    }
    noc_async_read_barrier();
    const int32_t s = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base)[0];
    const int32_t b0 = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + 64)[0];
    if constexpr (SHARD) {
        rank = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 128)[0];
        nchips = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 128)[1];
    }
    const uint32_t start = (s < 0 ? 0u : static_cast<uint32_t>(s)) + (b0 < 0 ? 0u : static_cast<uint32_t>(b0));
    const uint32_t l1_pools = base + 192;
    const uint32_t l1_out = l1_pools + POOLS_ALIGNED;
    const uint32_t l1_counts = l1_out + OUT_BYTES;
    volatile tt_l1_ptr uint32_t* pools = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_pools);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_out);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_counts);
    const InterleavedAddrGen<true> pools_g{.bank_base_address = pools_addr, .page_size = POOLS_BYTES};
    const InterleavedAddrGen<true> out_g{.bank_base_address = out_addr, .page_size = OUT_BYTES};
    const InterleavedAddrGen<true> counts_g{.bank_base_address = counts_addr, .page_size = COUNTS_BYTES};
    // SHARD: a global token id -> (owned here?, local row id)
    auto local_of = [&](uint32_t t) -> uint32_t { return (t / UNIT / nchips) * UNIT + t % UNIT; };
    auto owned = [&](uint32_t t) -> bool { return (t / UNIT) % nchips == rank; };

    for (uint32_t r = first; r < first + count; ++r) {
        noc_async_read(pools_g.get_noc_addr(r), l1_pools, POOLS_BYTES);
        noc_async_write_barrier();  // the previous row's ids (and counts) left the out buffers
        noc_async_read_barrier();
        const uint32_t p = start + r;
        const uint32_t n_closed = (p + 1) / KPOOL;
        const uint32_t nsel = n_closed < TOPK ? n_closed : TOPK;
        uint32_t i = 0;
        if constexpr (!SHARD) {
            for (uint32_t j = 0; j < nsel; ++j) {
                const uint32_t pool = pools[j];
                for (uint32_t k = 0; k < KPOOL; ++k) {
                    out[i++] = pool < n_closed ? KPOOL * pool + k : p;
                }
            }
            uint32_t t = KPOOL * n_closed;
            for (; i < WIDTH; ++i, ++t) {
                out[i] = t <= p ? t : 0xFFFFFFFFu;
            }
        } else {
            for (uint32_t j = 0; j < nsel; ++j) {
                const uint32_t pool = pools[j];
                if (pool < n_closed) {  // a whole pool is inside one unit: owned as one
                    if (owned(KPOOL * pool)) {
                        const uint32_t lp = (pool / POOL_PAGE / nchips) * POOL_PAGE + pool % POOL_PAGE;
                        for (uint32_t k = 0; k < KPOOL; ++k) {
                            out[i++] = KPOOL * lp + k;
                        }
                    }
                } else if (owned(p)) {  // the causal replacement key (the row's own position)
                    for (uint32_t k = 0; k < KPOOL; ++k) {
                        out[i++] = local_of(p);
                    }
                }
            }
            for (uint32_t t = KPOOL * n_closed; t <= p; ++t) {  // the open pool's tail
                if (owned(t)) {
                    out[i++] = local_of(t);
                }
            }
            cnt[0] = i;
            cnt[1] = i > 0 ? 0x3F800000u : 0u;  // the valid flag as fp32 bits (1.0f / 0.0f): the merge's mask
            for (uint32_t w = 2; w < COUNTS_BYTES / 4; ++w) {
                cnt[w] = 0;
            }
            if (i == 0) {
                out[i++] = 0;  // the dummy key (local row 0); the merge weighs this partial 0 from counts
            }
            for (; i < WIDTH; ++i) {
                out[i] = 0xFFFFFFFFu;
            }
            noc_async_write(l1_counts, counts_g.get_noc_addr(r), COUNTS_BYTES);
        }
        noc_async_write(l1_out, out_g.get_noc_addr(r), OUT_BYTES);
    }
    noc_async_write_barrier();
}
