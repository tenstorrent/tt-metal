// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 batched prefill (tt/prefill_tp.py): the request slot's KDA state between the decode stack's per-chunk state
// tensors and the prefill's per-layer scratch, INSIDE the prefill trace (the slot comes from device memory, so one
// captured program serves every slot). One core per KDA layer (BRISC).
//
//   S  : stack S of chunk c [rows <= chunk, H, 128, 128] fp32 TILE (user rd's tiles contiguous: pages [rd n_s, (rd + 1)
//   n_s);
//        the prefix pool's last chunk may hold fewer rows)
//        <-> scratch S [1, H, 128, 128] (n_s pages)
//   C  : stack conv history of chunk c [1, 3, rows <= chunk (32-row tiles), W] bf16 TILE (user rd = row rd of every
//   tile)
//        <-> scratch C [1, 3, 1, W] (row 0 of every tile, the other rows zero)
// STATE_IN  (define): scratch <- slot (fresh request: scratch <- 0)
// STATE_OUT (define): slot <- scratch (the slot's S pages; row rd of every conv tile, the other users' rows kept)
// STATE_COPY (define): slot word 0 <- slot word 2 directly (no scratch): the prefix cache's snapshot / restore between
//            a request slot and a snapshot pool entry (virtual slot), launched between steps, one core per layer
// Control tensor ctl [1, 32] uint32 ROW_MAJOR: word 0 = slot, word 1 = fresh (1: zero state), word 2 = the slot
// STATE_IN reads instead of word 0 (0xFFFFFFFF: word 0; a prefix-cache attach reads a snapshot pool entry = a
// virtual slot >= max_batch of the pool's extra state chunks), word 3 = a second slot STATE_OUT also writes
// (0xFFFFFFFF: none; a prefix-cache snapshot). Chunks n_chunks counts the pool's chunks.
// Compile-time: n_s (S pages per user), n_c (conv tiles), chunk (rows per state chunk), n_chunks.
// Runtime per core: ctl address, scratch S, scratch C, then (S_c, C_c) addresses of chunks 0 .. n_chunks - 1.
// CB SIO_CB: 2 x 4096 B + 2 x 2048 B staging.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t n_s = get_compile_time_arg_val(0);
    constexpr uint32_t n_c = get_compile_time_arg_val(1);
    constexpr uint32_t chunk = get_compile_time_arg_val(2);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t SB = 4096;   // fp32 tile
    constexpr uint32_t CB_ = 2048;  // bf16 tile
    const uint32_t ctl_addr = get_arg_val<uint32_t>(0);
    const uint32_t sc_s = get_arg_val<uint32_t>(1);
    const uint32_t sc_c = get_arg_val<uint32_t>(2);

    const uint32_t a = get_write_ptr(SIO_CB), b = a + SB, ca = b + SB, cb = ca + CB_;
    const InterleavedAddrGen<true> ctl_g{.bank_base_address = ctl_addr, .page_size = 32 * 4};
    noc_async_read(ctl_g.get_noc_addr(0), a, 32 * 4);
    noc_async_read_barrier();
    const uint32_t NONE = 0xFFFFFFFFu;
    const uint32_t dst_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a)[0];
    const uint32_t fresh = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a)[1];
    const uint32_t src_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a)[2];
    const uint32_t snap_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a)[3];
    const InterleavedAddrGen<true> s_scr{.bank_base_address = sc_s, .page_size = SB};
    const InterleavedAddrGen<true> c_scr{.bank_base_address = sc_c, .page_size = CB_};

#ifdef STATE_COPY
    (void)fresh;
    (void)snap_slot;
    const uint32_t s_slot_i = src_word, d_slot_i = dst_slot;
    const uint32_t sc = s_slot_i / chunk, sr = s_slot_i % chunk, dc = d_slot_i / chunk, dr = d_slot_i % chunk;
    const InterleavedAddrGen<true> s_src{.bank_base_address = get_arg_val<uint32_t>(3 + 2 * sc), .page_size = SB};
    const InterleavedAddrGen<true> s_dst{.bank_base_address = get_arg_val<uint32_t>(3 + 2 * dc), .page_size = SB};
    const InterleavedAddrGen<true> c_src{.bank_base_address = get_arg_val<uint32_t>(4 + 2 * sc), .page_size = CB_};
    const InterleavedAddrGen<true> c_dst{.bank_base_address = get_arg_val<uint32_t>(4 + 2 * dc), .page_size = CB_};
    (void)s_scr;
    (void)c_scr;
    for (uint32_t p = 0; p < n_s; p += 2) {
        noc_async_read(s_src.get_noc_addr(sr * n_s + p), a, SB);
        if (p + 1 < n_s) {
            noc_async_read(s_src.get_noc_addr(sr * n_s + p + 1), b, SB);
        }
        noc_async_read_barrier();
        noc_async_write(a, s_dst.get_noc_addr(dr * n_s + p), SB);
        if (p + 1 < n_s) {
            noc_async_write(b, s_dst.get_noc_addr(dr * n_s + p + 1), SB);
        }
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ca);
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb);
    for (uint32_t p = 0; p < n_c; ++p) {
        noc_async_read(c_src.get_noc_addr(p), ca, CB_);
        noc_async_read(c_dst.get_noc_addr(p), cb, CB_);
        noc_async_read_barrier();
        for (uint32_t c = 0; c < 32; ++c) {
            dst[tile_off(dr, c)] = src[tile_off(sr, c)];
        }
        noc_async_write(cb, c_dst.get_noc_addr(p), CB_);
        noc_async_write_barrier();
    }
#elif defined(STATE_IN)
    (void)snap_slot;
    const uint32_t slot = src_word == NONE ? dst_slot : src_word;
    const uint32_t cidx = slot / chunk, rd = slot % chunk;
    const uint32_t st_s = get_arg_val<uint32_t>(3 + 2 * cidx), st_c = get_arg_val<uint32_t>(4 + 2 * cidx);
    const InterleavedAddrGen<true> s_slot{.bank_base_address = st_s, .page_size = SB};
    const InterleavedAddrGen<true> c_slot{.bank_base_address = st_c, .page_size = CB_};
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    if (fresh) {
        for (uint32_t off = 0; off < SB; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, a + off, MEM_ZEROS_SIZE < SB - off ? MEM_ZEROS_SIZE : SB - off);
        }
        noc_async_read_barrier();
        for (uint32_t p = 0; p < n_s; ++p) {
            noc_async_write(a, s_scr.get_noc_addr(p), SB);
        }
        for (uint32_t p = 0; p < n_c; ++p) {
            noc_async_write(a, c_scr.get_noc_addr(p), CB_);
        }
        noc_async_write_barrier();
        return;
    }
    // S: slot pages -> scratch, two pages in flight
    for (uint32_t p = 0; p < n_s; p += 2) {
        noc_async_read(s_slot.get_noc_addr(rd * n_s + p), a, SB);
        if (p + 1 < n_s) {
            noc_async_read(s_slot.get_noc_addr(rd * n_s + p + 1), b, SB);
        }
        noc_async_read_barrier();
        noc_async_write(a, s_scr.get_noc_addr(p), SB);
        if (p + 1 < n_s) {
            noc_async_write(b, s_scr.get_noc_addr(p + 1), SB);
        }
        noc_async_write_barrier();
    }
    // C: row rd of every stack tile -> row 0 of a zero tile
    for (uint32_t off = 0; off < CB_; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, cb + off, MEM_ZEROS_SIZE < CB_ - off ? MEM_ZEROS_SIZE : CB_ - off);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ca);
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb);
    for (uint32_t p = 0; p < n_c; ++p) {
        noc_async_read(c_slot.get_noc_addr(p), ca, CB_);
        noc_async_read_barrier();
        for (uint32_t c = 0; c < 32; ++c) {
            dst[tile_off(0, c)] = src[tile_off(rd, c)];
        }
        noc_async_write(cb, c_scr.get_noc_addr(p), CB_);
        noc_async_write_barrier();
    }
#else  // STATE_OUT: the slot, then (prefix-cache snapshot) the second slot from the same scratch
    (void)fresh;
    (void)src_word;
    const uint32_t n_out = snap_slot == NONE ? 1u : 2u;
    for (uint32_t o = 0; o < n_out; ++o) {
        const uint32_t oslot = o == 0 ? dst_slot : snap_slot;
        const uint32_t ocidx = oslot / chunk, ord = oslot % chunk;
        const uint32_t os_s = get_arg_val<uint32_t>(3 + 2 * ocidx), os_c = get_arg_val<uint32_t>(4 + 2 * ocidx);
        const InterleavedAddrGen<true> so_slot{.bank_base_address = os_s, .page_size = SB};
        const InterleavedAddrGen<true> co_slot{.bank_base_address = os_c, .page_size = CB_};
        for (uint32_t p = 0; p < n_s; p += 2) {
            noc_async_read(s_scr.get_noc_addr(p), a, SB);
            if (p + 1 < n_s) {
                noc_async_read(s_scr.get_noc_addr(p + 1), b, SB);
            }
            noc_async_read_barrier();
            noc_async_write(a, so_slot.get_noc_addr(ord * n_s + p), SB);
            if (p + 1 < n_s) {
                noc_async_write(b, so_slot.get_noc_addr(ord * n_s + p + 1), SB);
            }
            noc_async_write_barrier();
        }
        volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ca);
        volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb);
        for (uint32_t p = 0; p < n_c; ++p) {
            noc_async_read(c_scr.get_noc_addr(p), ca, CB_);
            noc_async_read(co_slot.get_noc_addr(p), cb, CB_);
            noc_async_read_barrier();
            for (uint32_t c = 0; c < 32; ++c) {
                dst[tile_off(ord, c)] = src[tile_off(0, c)];
            }
            noc_async_write(cb, co_slot.get_noc_addr(p), CB_);
            noc_async_write_barrier();
        }
    }
#endif
}
