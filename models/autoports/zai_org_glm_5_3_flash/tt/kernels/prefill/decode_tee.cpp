// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14 prefix-cache G snapshots INSIDE the decode trace (tt/stack_model.py DecodeTee, GLM53_DECODE_TEE=1): the
// last program of every decode trace. For each pair (slot -> snapshot-pool entry, a virtual slot >= max_batch) listed
// in the device control tensor it copies the slot's POST-STEP state into the entry, so the entry is bitwise the state
// the continuing generation holds after that step (the step that consumes position 128 m - 1). One BRISC per layer.
//   TEE_KDA (one core per KDA layer, the state_io.cpp STATE_COPY body per pair):
//     S : stack S of chunk c [rows <= chunk, H, 128, 128] fp32 TILE (user r = pages [r n_s, (r + 1) n_s); the prefix
//     pool's
//         last chunk may hold fewer rows) -> the entry's pages
//     C : conv history of chunk c [1, 3, rows <= chunk, W] bf16 TILE (user r = row r of every tile) -> the entry's row
//     (read-
//         modify-write of the destination tile; the pairs run one after another on the core, so two entries sharing a
//         tile never race)
//   TEE_DSA (one core per sparse_dsa layer): the pooled-key store [slots Tp / 32, 1, 32, 128] bf16 (slot s = tiles
//     [s NT, (s + 1) NT), TPP tiles per 128-token page): tiles of pages [p0, p1) of the slot -> the entry; then the
//     slot's TR ring tiles [slots, 1, 32, 256] -> the entry's. Pages < p0 already sit in the entry (earlier crossings).
// ctl [1, 192] uint32 ROW_MAJOR (one 768-B page): word 0 = pairs n (<= 32); pair i at words 1 + 4 i .. 4 + 4 i =
//   (src slot, dst virtual slot, p0, p1). n = 0 on every step without a 128-boundary crossing: read ctl, return.
// Compile-time TEE_KDA: n_s (S pages per user), n_c (conv tiles per chunk), chunk, n_chunks, n_slots (max_batch + the
// pool's
//              exact entries: the watcher bound of every slot);
//              TEE_DSA: NT (store tiles per slot), TPP (store tiles per 128-token page), TR (ring tiles per slot).
// Runtime TEE_KDA: ctl address, then (S_c, C_c) addresses of chunks 0 .. n_chunks - 1 (the pool's chunks included);
//         TEE_DSA: ctl address, store address, ring address.
// CB TEE_CB: 2 x 4096 B + 2 x 2048 B staging, then the 768-B ctl copy.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

namespace {
constexpr uint32_t SB = 4096;   // fp32 tile
constexpr uint32_t CB_ = 2048;  // bf16 tile
constexpr uint32_t CTL_BYTES = 192 * 4;
constexpr uint32_t MAX_PAIRS = 32;

FORCE_INLINE uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

// pages [first, first + n) of src -> [dst_first, dst_first + n) of dst, two pages in flight
FORCE_INLINE void copy_pages(
    const InterleavedAddrGen<true>& src,
    uint32_t first,
    const InterleavedAddrGen<true>& dst,
    uint32_t dst_first,
    uint32_t n,
    uint32_t bytes,
    uint32_t a,
    uint32_t b) {
    for (uint32_t p = 0; p < n; p += 2) {
        noc_async_read(src.get_noc_addr(first + p), a, bytes);
        if (p + 1 < n) {
            noc_async_read(src.get_noc_addr(first + p + 1), b, bytes);
        }
        noc_async_read_barrier();
        noc_async_write(a, dst.get_noc_addr(dst_first + p), bytes);
        if (p + 1 < n) {
            noc_async_write(b, dst.get_noc_addr(dst_first + p + 1), bytes);
        }
        noc_async_write_barrier();
    }
}
}  // namespace

void kernel_main() {
    const uint32_t ctl_addr = get_arg_val<uint32_t>(0);
    const uint32_t a = get_write_ptr(TEE_CB), b = a + SB, ca = b + SB, cb = ca + CB_, ctl = cb + CB_;
    const InterleavedAddrGen<true> ctl_g{.bank_base_address = ctl_addr, .page_size = CTL_BYTES};
    noc_async_read(ctl_g.get_noc_addr(0), ctl, CTL_BYTES);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl);
    const uint32_t n_pairs = w[0];
    ASSERT(n_pairs <= MAX_PAIRS);
    if (n_pairs == 0 || n_pairs > MAX_PAIRS) {
        return;
    }
#if defined(TEE_KDA)
    constexpr uint32_t n_s = get_compile_time_arg_val(0);
    constexpr uint32_t n_c = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = get_compile_time_arg_val(2);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t n_slots = get_compile_time_arg_val(4);
    static_assert(n_slots <= chunk * n_chunks, "the slot bound lies inside the state chunks");
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ca);
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb);
    for (uint32_t i = 0; i < n_pairs; ++i) {
        const uint32_t s_slot = w[1 + 4 * i], d_slot = w[2 + 4 * i];
        ASSERT(s_slot < n_slots && d_slot < n_slots);
        const uint32_t sc = s_slot / chunk, sr = s_slot % chunk, dc = d_slot / chunk, dr = d_slot % chunk;
        const InterleavedAddrGen<true> s_src{.bank_base_address = get_arg_val<uint32_t>(1 + 2 * sc), .page_size = SB};
        const InterleavedAddrGen<true> s_dst{.bank_base_address = get_arg_val<uint32_t>(1 + 2 * dc), .page_size = SB};
        const InterleavedAddrGen<true> c_src{.bank_base_address = get_arg_val<uint32_t>(2 + 2 * sc), .page_size = CB_};
        const InterleavedAddrGen<true> c_dst{.bank_base_address = get_arg_val<uint32_t>(2 + 2 * dc), .page_size = CB_};
        copy_pages(s_src, sr * n_s, s_dst, dr * n_s, n_s, SB, a, b);
        for (uint32_t p = 0; p < n_c; ++p) {
            noc_async_read(c_src.get_noc_addr(p), ca, CB_);
            noc_async_read(c_dst.get_noc_addr(p), cb, CB_);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t c = 0; c < 32; ++c) {
                dst[tile_off(dr, c)] = src[tile_off(sr, c)];
            }
            noc_async_write(cb, c_dst.get_noc_addr(p), CB_);
            noc_async_write_barrier();
        }
    }
#elif defined(TEE_DSA)
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t TPP = get_compile_time_arg_val(1);
    constexpr uint32_t TR = get_compile_time_arg_val(2);
    const InterleavedAddrGen<true> store{.bank_base_address = get_arg_val<uint32_t>(1), .page_size = CB_};
    const InterleavedAddrGen<true> ring{.bank_base_address = get_arg_val<uint32_t>(2), .page_size = CB_};
    for (uint32_t i = 0; i < n_pairs; ++i) {
        const uint32_t s_slot = w[1 + 4 * i], d_slot = w[2 + 4 * i];
        const uint32_t p0 = w[3 + 4 * i], p1 = w[4 + 4 * i];
        ASSERT(p0 <= p1 && p1 * TPP <= NT);
        const uint32_t t0 = p0 * TPP, t1 = (p1 * TPP < NT ? p1 * TPP : NT);
        if (t1 > t0) {
            copy_pages(store, s_slot * NT + t0, store, d_slot * NT + t0, t1 - t0, CB_, a, b);
        }
        copy_pages(ring, s_slot * TR, ring, d_slot * TR, TR, CB_, a, b);
    }
#else
#error "decode_tee.cpp: define TEE_KDA or TEE_DSA"
#endif
}
