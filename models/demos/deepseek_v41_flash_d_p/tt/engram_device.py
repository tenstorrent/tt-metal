# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ENGRAM_DEVICE=on (P.87, owner 2026-11-19): the engram n-gram tables on the mesh, the hash and the row lookup on the
device, from the engine's device token ids (no id readback, no host table, no per-chunk upload of rows).

Tables (``DeviceTable``, uploaded ONCE a process, at the model's load): each engram layer's table as the packed
bfloat8_b rows of tt/engram_bfp8 (272 B a row: 16 shared exponents, 256 sign-magnitude bytes; the host path's rows
bit for bit), split over the mesh's chips by row ranges: chip q (row-major mesh order r TP + c) holds rows
[q Nc, (q + 1) Nc), Nc = ceil(N / chips), as int32 pages of 64 rows (17408 B; no padding). ~3.3 GB a chip a layer
on 8x4 (6.6 GB for layers 1 and 14). Built from the checkpoint (pread + tt/engram_bfp8.pack_into, the resident build's
code) or from the process's resident table, uploaded a block of 2^19 rows a chip at a time and copied into place by a
small DRAM copy program (ttnn has no partial host write).

Lookup (``EngramLookup``, one prepared program for the mesh, both engram layers, at begin_chunk). Per chip (66 cores):
  * hub (core (0, 0)): its mesh row's ids (the engine's [1, 1, W] uint32 shard, server order) into the chip's
    ids buffer, waits for the column peers' ids (fabric, an L1_SMALL counter), starts the hash cores, writes the
    chip's natural-order ids for the embedding (pad -> vocab - 1, as the host path's clamp), waits for every row of
    its own tokens, starts the convert cores.
  * hash cores (HASH_CORES): a slice of the chunk's natural positions each; compressed ids from the device token map,
    the tail from the slot's device id history (positions start - 3 .. start - 1, DEAD before 0), the n-gram hash
    (reference/ngram.EngramHasher bit for bit: wrapping int64 products, XOR, torch remainder, by Barrett reduction:
    the RISC-V cores have no 64-bit divide) only for the (n, head) sub-tables whose rows overlap this chip's range;
    every row in range becomes a work item (layer, local row, destination chip, staging row, sender). Each writes
    its slice's ids into the slot's id history.
  * senders: one core a first fabric hop (tt-metal's dimension-order routes, ttnn.bringup.combine_rf
    .sparse_combine_l1.route) and one for the chip's own tokens: the Y senders first send the chip's ids to its
    column peers; then every sender reads its items, reads each row from the chip's table shard (DRAM) and writes it
    to the owner chip's staging buffer (fabric: one packet a row with a fused increment of the owner's hub counter;
    own chip: a NoC write).
  * convert cores: the staged rows of the chip's own Q = chunk / chips tokens -> bfloat8_b TILE [1, 1, Q, 6144]
    (the HOST_HIDE=on upload's layout, gathered over TP by the model as before); rows past ``valid`` are zero.
Cross-chip counters are global semaphores taken back by their waiter (an early increment of the next call stays
counted); every wait is bounded (``status`` records a timeout instead of a hang).
"""

from __future__ import annotations

import os
import time

import torch

import ttnn

from .settings import settings

ROW = 272  # packed bfloat8_b table row
ROWS_PER_PAGE = 64
PAGE = ROW * ROWS_PER_PAGE  # 17408 B
PAGE_WORDS = PAGE // 4
STAGE = 320  # staging row stride (64 B aligned DRAM pages)
BLOCK_ROWS = 1 << 19  # upload block a chip
HASH_CORES = 32
CONVERT_CORES = 24
TMAP_PAGE = 16  # token map entries a 64 B page
HIST_PAGE = 32  # id history entries a 128 B page
DEAD = 0xFFFFFFFF
TIMEOUT = 1 << 22  # bounded waits (polls; well inside the dispatch timeout, the waits cascade)

_HEAD = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t TIMEOUT = %d;
inline bool wait_min(uint32_t addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < TIMEOUT; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return true;
        }
    }
    return false;
}
"""
    % TIMEOUT
)

# ---------------------------------------------------------------------------------------------------- hub
_HUB = (
    _HEAD
    + r"""
constexpr uint32_t W = get_compile_time_arg_val(0);      // ids a mesh row (a page of the ids tensors)
constexpr uint32_t SP = get_compile_time_arg_val(1);
constexpr uint32_t S = get_compile_time_arg_val(2);
constexpr uint32_t Q = get_compile_time_arg_val(3);      // tokens a chip
constexpr uint32_t NL = get_compile_time_arg_val(4);     // engram layers
constexpr uint32_t SLOTS = get_compile_time_arg_val(5);  // rows a token a layer (24)
constexpr uint32_t VOCAB = get_compile_time_arg_val(6);
constexpr uint32_t NCHIPS = get_compile_time_arg_val(7);
constexpr uint32_t PKT = get_compile_time_arg_val(8);
constexpr uint32_t META_P = get_compile_time_arg_val(9);
constexpr uint32_t REGION = get_compile_time_arg_val(10);
constexpr auto in_args = TensorAccessorArgs<11>();
constexpr auto all_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
constexpr auto emb_args = TensorAccessorArgs<all_args.next_compile_time_args_offset()>();
constexpr auto st_args = TensorAccessorArgs<emb_args.next_compile_time_args_offset()>();
constexpr auto tl_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
constexpr auto ib_args = TensorAccessorArgs<tl_args.next_compile_time_args_offset()>();
constexpr auto pt_args = TensorAccessorArgs<ib_args.next_compile_time_args_offset()>();

inline uint32_t server_index(uint32_t d, uint32_t start) {
    constexpr uint32_t w = W;
    const uint32_t intra = start % w;
    const uint32_t c0 = (start / w) % SP;
    if (d < w - intra) {
        return c0 * w + d;
    }
    if (d >= S - intra) {
        return c0 * w + (d - S + w);
    }
    const uint32_t k = (d + intra) / w;
    return ((c0 + k) % SP) * w + (d + intra - k * w);
}

void kernel_main() {
    size_t a = 0;
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t valid = get_arg_val<uint32_t>(a++);
    const uint32_t ids_in = get_arg_val<uint32_t>(a++);
    const uint32_t ids_all = get_arg_val<uint32_t>(a++);
    const uint32_t emb = get_arg_val<uint32_t>(a++);
    const uint32_t status = get_arg_val<uint32_t>(a++);
    const uint32_t tailp = get_arg_val<uint32_t>(a++);     // the slot's explicit tail [flag, c0, c1, c2]
    const uint32_t inbox = get_arg_val<uint32_t>(a++);
    const uint32_t ptab = get_arg_val<uint32_t>(a++);
    const uint32_t my_r = get_arg_val<uint32_t>(a++);      // mesh row
    const uint32_t my_q = get_arg_val<uint32_t>(a++);      // chip index
    const uint32_t ids_sem = get_arg_val<uint32_t>(a++);   // global: column peers' ids arrived
    const uint32_t rows_sem = get_arg_val<uint32_t>(a++);  // global: rows of my tokens arrived
    const uint32_t go_sem = get_arg_val<uint32_t>(a++);    // local semaphore id on hash / convert cores
    const uint32_t n_hash = get_arg_val<uint32_t>(a++);
    const size_t hash_xy = a;
    a += n_hash;
    const uint32_t n_conv = get_arg_val<uint32_t>(a++);
    const size_t conv_xy = a;
    a += n_conv;
    const auto IN = TensorAccessor(in_args, ids_in, W * 4);
    const auto ALL = TensorAccessor(all_args, ids_all, W * 4);
    const auto EMB = TensorAccessor(emb_args, emb, W * 4);
    const auto ST = TensorAccessor(st_args, status, 64);
    cb_reserve_back(1, 1);
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;  // S ids, 64 B status, W ids out, pointers, a meta page
    const uint32_t stat = buf + S * 4;
    const uint32_t ptr_l1 = (stat + 64 + W * 4 + 63) & ~63u;
    const uint32_t meta_l1 = ptr_l1 + NL * SLOTS * Q * 4;
    volatile tt_l1_ptr uint32_t* sv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stat);
    for (uint32_t i = 0; i < 16; ++i) {
        sv[i] = 0;
    }
    // my mesh row's ids -> ids_all page my_r (the peers' land there through the fabric)
    noc_async_read(IN.get_noc_addr(0), buf, W * 4);
    noc_async_read_barrier();
    noc_async_write(buf, ALL.get_noc_addr(my_r), W * 4);
    noc_async_write_barrier();
    const uint64_t ids_noc = get_noc_addr(ids_sem);
    if (SP > 1) {
        if (!wait_min(ids_sem, SP - 1)) {
            sv[0] = 1;
        } else {
            noc_semaphore_inc(ids_noc, 0u - (SP - 1));
            noc_async_atomic_barrier();
        }
    }
    const uint32_t go = get_semaphore(go_sem);
    for (uint32_t i = 0; i < n_hash; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(hash_xy + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, go), 1);
    }
    noc_async_atomic_barrier();
    // natural-order ids of my mesh row for the embedding (positions >= valid: the clamped pad, VOCAB - 1)
    for (uint32_t r = 0; r < SP; ++r) {
        noc_async_read(ALL.get_noc_addr(r), buf + r * W * 4, W * 4);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t out = buf + S * 4 + 64;
    volatile tt_l1_ptr uint32_t* ov = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    for (uint32_t j = 0; j < W; ++j) {
        const uint32_t d = my_r * W + j;
        uint32_t id = d < valid ? ids[server_index(d, start)] : VOCAB - 1;
        ov[j] = id < VOCAB ? id : VOCAB - 1;
    }
    noc_async_write(out, EMB.get_noc_addr(0), W * 4);
    // every chip's rows of my tokens (its meta packet, counted, closes its inbox region of mine), then the pointer
    // table [layer][slot][token] -> (source chip, entry, the row's 16 B offset) in my L1 for the convert cores
    const uint32_t expect = DIAG ? 0 : NCHIPS;
    if (expect) {
        if (!wait_min(rows_sem, expect)) {
            sv[1] = 1;
            sv[2] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_sem);
            sv[3] = expect;
        } else {
            noc_semaphore_inc(get_noc_addr(rows_sem), 0u - expect);
            noc_async_atomic_barrier();
            const auto IB = TensorAccessor(ib_args, inbox, PKT);
            volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptr_l1);
            volatile tt_l1_ptr uint32_t* mw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
            uint32_t total = 0;
            for (uint32_t src = 0; src < NCHIPS; ++src) {
                noc_async_read(IB.get_noc_addr(src * REGION), meta_l1, PKT);
                noc_async_read_barrier();
                uint32_t cnt = mw[0];
                if (cnt > NL * SLOTS * Q) {  // (never: a bad count reads no further)
                    sv[4] = cnt;
                    cnt = 0;
                }
                total += cnt;
                uint32_t pg = 0;
                for (uint32_t j = 0; j < cnt; ++j) {
                    const uint32_t wi = 1 + j;
                    if (wi / (PKT / 4) != pg) {
                        pg = wi / (PKT / 4);
                        noc_async_read(IB.get_noc_addr(src * REGION + pg), meta_l1, PKT);
                        noc_async_read_barrier();
                    }
                    const uint32_t w = mw[wi % (PKT / 4)];
                    const uint32_t idx = w & 0xFFFF;
                    const uint32_t l = w >> 31;
                    if (idx >= SLOTS * Q || l >= NL) {  // (never: a bad word writes nothing)
                        sv[5] = w;
                        continue;
                    }
                    ptr[(l * SLOTS + idx % SLOTS) * Q + idx / SLOTS] = (src << 20) | (j << 2) | ((w >> 29) & 3u);
                }
            }
            const uint32_t lo = my_q * Q;
            const uint32_t mine = valid <= lo ? 0 : ((valid - lo) < Q ? (valid - lo) : Q);
            if (total != mine * SLOTS * NL) {
                sv[1] = 2;
                sv[2] = total;
                sv[3] = mine * SLOTS * NL;
            }
        }
    }
    {  // the pointer table to DRAM (pages of 32 pointers: one convert unit's), then the convert cores start
        const auto PT = TensorAccessor(pt_args, ptab, 128);
        for (uint32_t pg = 0; pg < NL * SLOTS * Q / 32; ++pg) {
            noc_async_write(ptr_l1 + pg * 128, PT.get_noc_addr(pg), 128);
        }
        noc_async_write_barrier();
    }
    for (uint32_t i = 0; i < n_conv; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(conv_xy + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, go), 1);
    }
    noc_async_atomic_barrier();
    noc_async_write(stat, ST.get_noc_addr(0), 64);
    // the explicit tail served this chunk only: cleared once every hash core is done (they all are: rows arrived)
    const auto TL = TensorAccessor(tl_args, tailp, 64);
    for (uint32_t i = 0; i < 16; ++i) {
        ids[i] = 0;  // (the ids area is free now; ``out`` may still be on its way to the embedding's ids)
    }
    noc_async_write(buf, TL.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
"""
)

# ---------------------------------------------------------------------------------------------------- hash
_HASH = (
    _HEAD
    + r"""
constexpr uint32_t W = get_compile_time_arg_val(0);
constexpr uint32_t SP = get_compile_time_arg_val(1);
constexpr uint32_t S = get_compile_time_arg_val(2);
constexpr uint32_t Q = get_compile_time_arg_val(3);
constexpr uint32_t NL = get_compile_time_arg_val(4);
constexpr uint32_t HEADS = get_compile_time_arg_val(5);
constexpr uint32_t VOCAB = get_compile_time_arg_val(6);
constexpr uint32_t PAD = get_compile_time_arg_val(7);   // compressed id of the pad token
constexpr uint32_t CAP = get_compile_time_arg_val(8);   // items this core can hold
constexpr auto all_args = TensorAccessorArgs<9>();
constexpr auto hist_args = TensorAccessorArgs<all_args.next_compile_time_args_offset()>();
constexpr auto map_args = TensorAccessorArgs<hist_args.next_compile_time_args_offset()>();
constexpr auto tl_args = TensorAccessorArgs<map_args.next_compile_time_args_offset()>();
constexpr uint32_t DEADC = 0xFFFFFFFFu;
constexpr uint32_t SLOTS = 3 * HEADS;

inline uint32_t server_index(uint32_t d, uint32_t start) {
    constexpr uint32_t w = W;
    const uint32_t intra = start % w;
    const uint32_t c0 = (start / w) % SP;
    if (d < w - intra) {
        return c0 * w + d;
    }
    if (d >= S - intra) {
        return c0 * w + (d - S + w);
    }
    const uint32_t k = (d + intra) / w;
    return ((c0 + k) % SP) * w + (d + intra - k * w);
}

inline uint64_t mul64(uint64_t a, uint64_t b) { return a * b; }  // low 64 bits (wraps as torch's int64 multiply)

inline uint64_t mulhi64(uint64_t a, uint64_t b) {
    const uint64_t a0 = static_cast<uint32_t>(a), a1 = a >> 32, b0 = static_cast<uint32_t>(b), b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// x mod p (x < 2^64) by Barrett: m = floor((2^64 - 1) / p)
inline uint32_t umod(uint64_t x, uint32_t p, uint64_t m) {
    uint64_t r = x - mulhi64(x, m) * p;
    while (r >= p) {
        r -= p;
    }
    return static_cast<uint32_t>(r);
}

// torch.remainder of the int64 x by p > 0
inline uint32_t floormod(uint64_t x, uint32_t p, uint64_t m) {
    if (static_cast<int64_t>(x) >= 0) {
        return umod(x, p, m);
    }
    const uint32_t r = umod(~x + 1, p, m);
    return r ? p - r : 0;
}

void kernel_main() {
    size_t a = 0;
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t valid = get_arg_val<uint32_t>(a++);
    const uint32_t ids_all = get_arg_val<uint32_t>(a++);
    const uint32_t hist = get_arg_val<uint32_t>(a++);
    const uint32_t tmap = get_arg_val<uint32_t>(a++);
    const uint32_t tailp = get_arg_val<uint32_t>(a++);
    const uint32_t lo = get_arg_val<uint32_t>(a++);  // my natural positions [lo, hi)
    const uint32_t hi = get_arg_val<uint32_t>(a++);
    const uint32_t go_sem = get_arg_val<uint32_t>(a++);
    const uint32_t done_sem = get_arg_val<uint32_t>(a++);
    const uint32_t n_send = get_arg_val<uint32_t>(a++);
    const size_t send_xy = a;
    a += n_send;
    const size_t dest_sender = a;  // a word a destination chip: its sender index
    a += get_arg_val<uint32_t>(a) + 1;
    // per layer: 8 words of multipliers (lo, hi x 4), base row, owned count, then per owned: slot, p, m_lo, m_hi, off
    const size_t layers = a;
    const auto ALL = TensorAccessor(all_args, ids_all, W * 4);
    const auto HIST = TensorAccessor(hist_args, hist, HIST_PAGE_BYTES);
    const auto MAP = TensorAccessor(map_args, tmap, 64);
    cb_reserve_back(1, 1);
    const uint32_t ids_l1 = (get_write_ptr(1) + 63) & ~63u;  // S ids (server order)
    const uint32_t comp_l1 = ids_l1 + S * 4;                  // comp ids of [lo - 3, hi)
    // natural ids of [lo, hi) (the history write: a DRAM write's L1 source at the page's 64 B alignment)
    const uint32_t nat_l1 = (comp_l1 + (hi - lo + 3) * 4 + 63) & ~63u;
    const uint32_t rd_l1 = (nat_l1 + (hi - lo) * 4 + 64 + 63) & ~63u;  // 64 x 64 B token-map pages / history page
    const uint32_t items = get_write_ptr(0);            // [count, pad x 3, items x 2 words]
    volatile tt_l1_ptr uint32_t* it = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(items);
    it[0] = 0;
    const uint32_t go = get_semaphore(go_sem);
    bool ok = wait_min(go, 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go), 0);
    uint32_t n_items = 0;
    const uint32_t end = hi < valid ? hi : valid;
    if (ok && lo < end) {
        for (uint32_t r = 0; r < SP; ++r) {
            noc_async_read(ALL.get_noc_addr(r), ids_l1 + r * W * 4, W * 4);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
        volatile tt_l1_ptr uint32_t* comp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(comp_l1);
        volatile tt_l1_ptr uint32_t* nat = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(nat_l1);
        // the 3 positions before lo: from the chunk, else the slot's explicit tail (compressed ids, flag set), else
        // the slot's id history, else DEAD
        uint32_t premapped = 0;
        if (lo == 0) {
            const auto TL = TensorAccessor(tl_args, tailp, 64);
            noc_async_read(TL.get_noc_addr(0), rd_l1, 64);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* tv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd_l1);
            if (tv[0]) {
                premapped = 3;
                comp[0] = tv[1];
                comp[1] = tv[2];
                comp[2] = tv[3];
            }
        }
        for (uint32_t k = premapped; k < 3; ++k) {
            const int32_t d = static_cast<int32_t>(lo) - 3 + static_cast<int32_t>(k);
            const int32_t pos = static_cast<int32_t>(start) + d;
            uint32_t id = DEADC;
            if (d >= 0) {
                id = ids[server_index(static_cast<uint32_t>(d), start)];
            } else if (pos >= 0) {
                noc_async_read(HIST.get_noc_addr(static_cast<uint32_t>(pos) / HIST_PAGE, 0), rd_l1, HIST_PAGE_BYTES);
                noc_async_read_barrier();
                id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd_l1)[static_cast<uint32_t>(pos) % HIST_PAGE];
            }
            comp[k] = id;  // (an id for now; mapped below)
        }
        for (uint32_t d = lo; d < end; ++d) {
            const uint32_t id = ids[server_index(d, start)];
            nat[d - lo] = id;
            comp[3 + d - lo] = id;
        }
        // ids -> compressed ids through the token map (64 B pages, 64 reads a barrier)
        const uint32_t n = 3 + end - lo;
        for (uint32_t b0 = premapped; b0 < n; b0 += 64) {
            const uint32_t nb = (n - b0) < 64 ? (n - b0) : 64;
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t id = comp[b0 + i];
                if (id < VOCAB) {
                    noc_async_read(MAP.get_noc_addr(id / TMAP_PAGE), rd_l1 + i * 64, 64);
                }
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t id = comp[b0 + i];
                comp[b0 + i] = id < VOCAB ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd_l1 + i * 64)[id % TMAP_PAGE]
                                          : DEADC;
            }
        }
        // the hash, layer by layer, only the sub-tables overlapping my rows
        for (uint32_t t = lo; t < end; ++t) {
            uint32_t tok[4];
            bool blocked = false;
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t c = comp[3 + t - lo - s];
                if (c == DEADC) {
                    blocked = true;
                }
                tok[s] = blocked ? PAD : c;
            }
            const uint32_t dq = t / Q;
            const uint32_t sender = get_arg_val<uint32_t>(dest_sender + 1 + dq);
            const uint32_t idx0 = (t % Q) * SLOTS;
            size_t la = layers;
            for (uint32_t l = 0; l < NL; ++l) {
                uint64_t roll[3];
                uint64_t r = 0;
                for (uint32_t s = 0; s < 4; ++s) {
                    const uint64_t mult = static_cast<uint64_t>(get_arg_val<uint32_t>(la + 2 * s)) |
                                          (static_cast<uint64_t>(get_arg_val<uint32_t>(la + 2 * s + 1)) << 32);
                    const uint64_t prod = mul64(static_cast<uint64_t>(tok[s]), mult);
                    r = s == 0 ? prod : (r ^ prod);
                    if (s > 0) {
                        roll[s - 1] = r;
                    }
                }
                const uint32_t base = get_arg_val<uint32_t>(la + 8);
                const uint32_t nc = get_arg_val<uint32_t>(la + 9);
                const uint32_t nown = get_arg_val<uint32_t>(la + 10);
                for (uint32_t o = 0; o < nown; ++o) {
                    const size_t e = la + 11 + 5 * o;
                    const uint32_t slot = get_arg_val<uint32_t>(e);
                    const uint32_t p = get_arg_val<uint32_t>(e + 1);
                    const uint64_t m = static_cast<uint64_t>(get_arg_val<uint32_t>(e + 2)) |
                                       (static_cast<uint64_t>(get_arg_val<uint32_t>(e + 3)) << 32);
                    const uint32_t row = floormod(roll[slot / HEADS], p, m) + get_arg_val<uint32_t>(e + 4);
                    const uint32_t lr = row - base;  // (wraps past the range: one compare)
                    if (lr < nc) {
                        if (n_items < CAP) {
                            it[4 + 2 * n_items] = (l << 31) | lr;
                            it[4 + 2 * n_items + 1] = (sender << 28) | (dq << 16) | (idx0 + slot);
                        }
                        ++n_items;
                    }
                }
                la += 11 + 5 * nown;
            }
        }
        // my slice's ids -> the slot's id history [start + lo, start + end) (whole 32-id pages; the last one merged)
        const uint32_t p0 = (start + lo) / HIST_PAGE;
        const uint32_t full = (end - lo) / HIST_PAGE;
        for (uint32_t k = 0; k < full; ++k) {
            noc_async_write(nat_l1 + k * HIST_PAGE_BYTES, HIST.get_noc_addr(p0 + k, 0), HIST_PAGE_BYTES);
        }
        const uint32_t rest = (end - lo) - full * HIST_PAGE;
        if (rest) {
            noc_async_read(HIST.get_noc_addr(p0 + full, 0), rd_l1, HIST_PAGE_BYTES);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* pg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd_l1);
            for (uint32_t i = 0; i < rest; ++i) {
                pg[i] = nat[full * HIST_PAGE + i];
            }
            noc_async_write(rd_l1, HIST.get_noc_addr(p0 + full, 0), HIST_PAGE_BYTES);
        }
        noc_async_write_barrier();
    }
    it[0] = n_items < CAP ? n_items : CAP;
    it[1] = n_items;  // (more than CAP: the overflow is visible to the senders' status)
    const uint32_t done = get_semaphore(done_sem);
    for (uint32_t i = 0; i < n_send; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(send_xy + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, done), 1);
    }
    noc_async_atomic_barrier();
}
"""
)

# ---------------------------------------------------------------------------------------------------- senders
_SEND = (
    _HEAD
    + r"""
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
constexpr uint32_t W = get_compile_time_arg_val(0);
constexpr uint32_t NL = get_compile_time_arg_val(1);
constexpr uint32_t FABRIC = get_compile_time_arg_val(2);
constexpr uint32_t IDS_SEG = get_compile_time_arg_val(3);  // bytes an ids packet
constexpr uint32_t B = get_compile_time_arg_val(4);        // rows a packet (RPP)
constexpr uint32_t RPP = B;
constexpr uint32_t PKT = get_compile_time_arg_val(5);      // inbox page = a full packet (RPP windows)
constexpr uint32_t META_P = get_compile_time_arg_val(6);   // meta pages a source region
constexpr uint32_t REGION = get_compile_time_arg_val(7);   // pages a source region (meta + entries)
constexpr auto in_args = TensorAccessorArgs<8>();
constexpr auto all_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
constexpr auto t0_args = TensorAccessorArgs<all_args.next_compile_time_args_offset()>();
constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
constexpr auto ib_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();

inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint32_t node) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(node & 0xFFFF), static_cast<uint16_t>(node >> 16));
#else
#error "the engram exchange needs a 2D fabric (agent rule 4)"
#endif
}

void kernel_main() {
    size_t a = 0;
    const uint32_t ids_in = get_arg_val<uint32_t>(a++);
    const uint32_t ids_all = get_arg_val<uint32_t>(a++);
    const uint32_t tab0 = get_arg_val<uint32_t>(a++);
    const uint32_t tab1 = get_arg_val<uint32_t>(a++);
    const uint32_t inbox = get_arg_val<uint32_t>(a++);
    const uint32_t my_q = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);       // my sender index
    const uint32_t my_r = get_arg_val<uint32_t>(a++);
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint32_t ids_sem = get_arg_val<uint32_t>(a++);
    const uint32_t rows_sem = get_arg_val<uint32_t>(a++);
    const uint32_t done_sem = get_arg_val<uint32_t>(a++);
    const uint32_t n_hash = get_arg_val<uint32_t>(a++);
    const size_t hash_xy = a;
    a += n_hash;
    const uint32_t n_node = get_arg_val<uint32_t>(a++);  // fabric node of every chip (mesh << 16 | chip)
    const size_t nodes = a;
    a += n_node;
    const uint32_t n_peer = get_arg_val<uint32_t>(a++);  // column peers I send my ids to (chip indices)
    const size_t peers = a;
    a += n_peer;
    const uint32_t n_mine = get_arg_val<uint32_t>(a++);  // destination chips I serve
    const size_t mine = a;
    a += n_mine;
    const auto IN = TensorAccessor(in_args, ids_in, W * 4);
    const auto ALL = TensorAccessor(all_args, ids_all, W * 4);
    const auto T0 = TensorAccessor(t0_args, tab0, ROW_PAGE);
    const auto T1 = TensorAccessor(t1_args, tab1, ROW_PAGE);
    cb_reserve_back(1, 1);
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;  // B row slots of STAGE_BYTES, the item window, B headers
    const uint32_t win = buf + 2 * B * STAGE_BYTES;  // (two halves of B windows: one read while the other is sent)
    const uint32_t hdrs = win + ITEM_WIN * 8;  // B packet headers of HDR_SLOT bytes (one flush a batch, not a row)
    for (uint32_t i = 0; i < B * HDR_SLOT / 4; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdrs)[i] = 0;
    }
    const uint32_t hx = hub_xy >> 16, hy = hub_xy & 0xFFFF;
    const uint64_t rows_noc = get_noc_addr(hx, hy, rows_sem);
    const uint32_t items_base = get_write_ptr(0);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = nullptr;
    WorkerToFabricEdmSender conn;
    if constexpr (FABRIC) {
        conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        hdr = PacketHeaderPool::allocate_header();
        conn.open();
        // my mesh row's ids to the column peers whose route starts on my port
        if (n_peer) {
            noc_async_read(IN.get_noc_addr(0), buf, W * 4);
            noc_async_read_barrier();
            const uint64_t ids_noc = get_noc_addr(hx, hy, ids_sem);
            for (uint32_t i = 0; i < n_peer; ++i) {
                route_to(hdr, get_arg_val<uint32_t>(nodes + get_arg_val<uint32_t>(peers + i)));
                for (uint32_t off = 0; off < W * 4; off += IDS_SEG) {
                    const uint32_t bytes = (W * 4 - off) < IDS_SEG ? (W * 4 - off) : IDS_SEG;
                    const uint64_t dst = ALL.get_noc_addr(my_r, off);
                    if (off + bytes >= W * 4) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, ids_noc, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(buf + off, bytes, reinterpret_cast<uint32_t>(hdr));
                    noc_async_writes_flushed();
                }
            }
        }
    }
    const uint32_t done = get_semaphore(done_sem);
    const bool ok = wait_min(done, n_hash);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done), 0);
    // rows -> the owner chip's inbox region of this chip: entries (STAGE_BYTES windows, the row at its 16 B offset)
    // in packets of RPP, then the index words (meta) with the arrival increment; my destinations in item order
    // (ascending: the hash cores' slices are in token order), every one of mine gets its meta (count 0 too)
    const auto IB = TensorAccessor(ib_args, inbox, PKT);
    const uint32_t meta = (hdrs + B * HDR_SLOT + 63) & ~63u;  // meta words of the current destination: [count, words...]
    volatile tt_l1_ptr uint32_t* mv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    uint32_t cur = 0xFFFFFFFFu, k = 0, nb = 0, fin = 0, half = 0;
    auto send = [&](uint32_t d, uint32_t page, uint32_t src, uint32_t bytes, bool last) {
        const uint64_t dst = IB.get_noc_addr(page);
        if constexpr (FABRIC) {
            noc_async_writes_flushed();  // (the header's previous packet left L1 before it is rewritten)
            route_to(hdr, get_arg_val<uint32_t>(nodes + d));
            if (last) {
                hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, rows_noc, 1, true},
                                                           bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(src, bytes, reinterpret_cast<uint32_t>(hdr));
        } else {
            noc_async_write(src, dst, bytes);
            noc_async_write_barrier();
            if (last) {
                noc_semaphore_inc(rows_noc, 1);
                noc_async_atomic_barrier();
            }
        }
    };
    auto flush = [&]() {
        if (nb == 0) {
            return;
        }
        noc_async_read_barrier();
#if DIAG != 2
        send(cur, my_q * REGION + META_P + k / RPP, buf + half * B * STAGE_BYTES, nb * STAGE_BYTES, false);
#endif
        k += nb;
        nb = 0;
        half ^= 1u;  // (send's flush before its next packet waits for this one: this half is reread two batches on)
    };
    auto finish = [&](uint32_t d) {  // the meta of destination d (my rows to it: k), the last packet counted
        mv[0] = k;
        const uint32_t bytes = ((k + 1) * 4 + 63) & ~63u;  // (whole 64 B lines: no partial-line DRAM write)
        for (uint32_t off = 0; off < bytes; off += PKT) {
            const uint32_t n = (bytes - off) < PKT ? (bytes - off) : PKT;
            send(d, my_q * REGION + off / PKT, meta + off, n, off + n >= bytes);
        }
        fin |= 1u << d;
        noc_async_writes_flushed();  // (the meta words left L1 before the next destination's are written)
    };
    if (ok && DIAG != 1) {
        for (uint32_t h = 0; h < n_hash; ++h) {
            const uint32_t xy = get_arg_val<uint32_t>(hash_xy + h);
            const uint64_t src_items = get_noc_addr(xy >> 16, xy & 0xFFFF, items_base);
            noc_async_read(src_items, win, 16);
            noc_async_read_barrier();
            const uint32_t cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(win)[0];
            for (uint32_t i0 = 0; i0 < cnt; i0 += ITEM_WIN) {
                const uint32_t ni = (cnt - i0) < ITEM_WIN ? (cnt - i0) : ITEM_WIN;
                noc_async_read(src_items + 16 + i0 * 8, win, ni * 8);
                noc_async_read_barrier();
                volatile tt_l1_ptr uint32_t* iv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(win);
                for (uint32_t i = 0; i < ni; ++i) {
                    const uint32_t w1 = iv[2 * i + 1];
                    if ((w1 >> 28) != me) {
                        continue;
                    }
                    const uint32_t d = (w1 >> 16) & 0xFFF;
                    if (d != cur) {
                        flush();
                        if (cur != 0xFFFFFFFFu) {
                            finish(cur);
                        }
                        cur = d;
                        k = 0;
                    }
                    const uint32_t w0 = iv[2 * i];
                    const uint32_t lr = w0 & 0x7FFFFFFFu;
                    const uint32_t off = (lr % ROWS_PER_PAGE) * ROW_BYTES;
                    const uint64_t s = (w0 >> 31) ? T1.get_noc_addr(lr / ROWS_PER_PAGE, off & ~63u)
                                                  : T0.get_noc_addr(lr / ROWS_PER_PAGE, off & ~63u);
                    noc_async_read(s, buf + (half * B + nb) * STAGE_BYTES, STAGE_BYTES);
                    // meta word: layer, the row's 16 B offset in its window, staging index (token x slots + slot)
                    mv[1 + k + nb] = (w0 & 0x80000000u) | (((off & 63u) >> 4) << 29) | (w1 & 0xFFFF);
                    if (++nb == RPP) {
                        flush();
                    }
                }
            }
        }
        flush();
        if (cur != 0xFFFFFFFFu) {
            finish(cur);
        }
    }
    // my destinations that got no row: their meta (count 0) and increment all the same
    k = 0;
    for (uint32_t j = 0; j < n_mine && DIAG != 1; ++j) {
        const uint32_t d = get_arg_val<uint32_t>(mine + j);
        if (!((fin >> d) & 1u)) {
            finish(d);
        }
    }
    if constexpr (FABRIC) {
        conn.close();
    }
}
"""
)

# ---------------------------------------------------------------------------------------------------- convert
_CONV = (
    _HEAD
    + r"""
constexpr uint32_t Q = get_compile_time_arg_val(0);
constexpr uint32_t SLOTS = get_compile_time_arg_val(1);
constexpr uint32_t NL = get_compile_time_arg_val(2);
constexpr uint32_t PKT = get_compile_time_arg_val(3);
constexpr uint32_t META_P = get_compile_time_arg_val(4);
constexpr uint32_t REGION = get_compile_time_arg_val(5);
constexpr uint32_t RPP = PKT / STAGE_BYTES;
constexpr uint32_t NCHIPS = get_compile_time_arg_val(6);
constexpr auto ib_args = TensorAccessorArgs<7>();
constexpr auto pt_args = TensorAccessorArgs<ib_args.next_compile_time_args_offset()>();
constexpr auto o0_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
constexpr auto o1_args = TensorAccessorArgs<o0_args.next_compile_time_args_offset()>();
constexpr uint32_t TCOLS = SLOTS * 8;  // tiles across a tile row (6144 / 32)

void kernel_main() {
    size_t a = 0;
    const uint32_t start = get_arg_val<uint32_t>(a++);
    const uint32_t valid = get_arg_val<uint32_t>(a++);
    const uint32_t inbox = get_arg_val<uint32_t>(a++);
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint32_t ptab = get_arg_val<uint32_t>(a++);  // the hub's pointer table (DRAM, 128 B pages)
    const uint32_t out0 = get_arg_val<uint32_t>(a++);
    const uint32_t out1 = get_arg_val<uint32_t>(a++);
    const uint32_t my_q = get_arg_val<uint32_t>(a++);
    const uint32_t go_sem = get_arg_val<uint32_t>(a++);
    const uint32_t u0 = get_arg_val<uint32_t>(a++);  // my units [u0, u1) of (layer, tile row, slot)
    const uint32_t u1 = get_arg_val<uint32_t>(a++);
    (void)start;
    const auto IB = TensorAccessor(ib_args, inbox, PKT);
    const auto O0 = TensorAccessor(o0_args, out0, 1088);
    const auto O1 = TensorAccessor(o1_args, out1, 1088);
    cb_reserve_back(1, 1);
    const uint32_t rows = (get_write_ptr(1) + 63) & ~63u;  // 32 staged rows
    const uint32_t tiles = rows + 32 * STAGE_BYTES;        // 8 tiles
    const uint32_t ptrs = tiles + 8 * 1088;                // the unit's 32 pointers
    const uint32_t go = get_semaphore(go_sem);
    const bool ok = wait_min(go, 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go), 0);
    if (!ok) {
        return;
    }
    constexpr uint32_t TR = Q / 32;
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t l = u / (TR * SLOTS);
        const uint32_t tr = (u / SLOTS) % TR;
        const uint32_t slot = u % SLOTS;
        const uint32_t t0 = my_q * Q + tr * 32;  // the tile row's first token (chunk offset)
        const uint32_t live = valid <= t0 ? 0 : ((valid - t0) < 32 ? (valid - t0) : 32);
        uint32_t delta[32];
        if (live) {
            const auto PT = TensorAccessor(pt_args, ptab, 128);
            noc_async_read(PT.get_noc_addr(((l * SLOTS + slot) * Q + tr * 32) / 32), ptrs, 128);
            noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint32_t* pv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptrs);
        for (uint32_t i = 0; i < live; ++i) {
            const uint32_t pw = pv[i];
            const uint32_t src = pw >> 20, j = (pw >> 2) & 0x3FFFF;
            delta[i] = (pw & 3u) * 16;
            if (src >= NCHIPS || j >= (REGION - META_P) * RPP) {  // (never: a bad pointer reads nothing)
                continue;
            }
            noc_async_read(IB.get_noc_addr(src * REGION + META_P + j / RPP, (j % RPP) * STAGE_BYTES), rows + i * STAGE_BYTES,
                           STAGE_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t tb = tiles + k * 1088;
            volatile tt_l1_ptr uint8_t* ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tb);
            for (uint32_t f = 0; f < 4; ++f) {
                const uint32_t fr = f >> 1, fc = f & 1;
                for (uint32_t rr = 0; rr < 16; ++rr) {
                    const uint32_t i = fr * 16 + rr;
                    volatile tt_l1_ptr uint32_t* dm =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tb + 64 + f * 256 + rr * 16);
                    if (i < live) {
                        const uint32_t rb = rows + i * STAGE_BYTES + delta[i];
                        ex[f * 16 + rr] = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(rb)[2 * k + fc];
                        volatile tt_l1_ptr uint32_t* sm =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rb + 16 + 32 * k + 16 * fc);
                        dm[0] = sm[0];
                        dm[1] = sm[1];
                        dm[2] = sm[2];
                        dm[3] = sm[3];
                    } else {
                        ex[f * 16 + rr] = 0;
                        dm[0] = 0;
                        dm[1] = 0;
                        dm[2] = 0;
                        dm[3] = 0;
                    }
                }
            }
            const uint32_t page = tr * TCOLS + slot * 8 + k;
            noc_async_write(tb, l ? O1.get_noc_addr(page) : O0.get_noc_addr(page), 1088);
        }
        noc_async_write_barrier();
    }
}
"""
)

# ---------------------------------------------------------------------------------------------------- table copy
_COPY = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr auto s_args = TensorAccessorArgs<0>();
constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t lo = get_arg_val<uint32_t>(2);
    const uint32_t hi = get_arg_val<uint32_t>(3);
    const uint32_t p0 = get_arg_val<uint32_t>(4);
    const auto S = TensorAccessor(s_args, src, ROW_PAGE);
    const auto D = TensorAccessor(d_args, dst, ROW_PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    for (uint32_t p = lo; p < hi; ++p) {
        noc_async_read(S.get_noc_addr(p), buf, ROW_PAGE);
        noc_async_read_barrier();
        noc_async_write(buf, D.get_noc_addr(p0 + p), ROW_PAGE);
        noc_async_write_barrier();
    }
}
"""


def _defines(src: str) -> str:
    """Kernel constants shared with the host."""
    pre = (
        f"#define ROW_BYTES {ROW}u\n#define ROWS_PER_PAGE {ROWS_PER_PAGE}u\n#define ROW_PAGE {PAGE}u\n"
        f"#define STAGE_BYTES {STAGE}u\n#define TMAP_PAGE {TMAP_PAGE}u\n#define HIST_PAGE {HIST_PAGE}u\n"
        f"#define HIST_PAGE_BYTES {HIST_PAGE * 4}u\n#define ITEM_WIN 256u\n#define HDR_SLOT 160u\n"
        # diagnostic (P.87 timing split, never a shipped path; the P87_DIAG Setting): 1 = no row reads / sends, 2 = reads
        f"#define DIAG {int(settings.get('P87_DIAG'))}\n"
    )
    return pre + src


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _xy(mesh, cc) -> int:
    v = mesh.worker_core_from_logical_core(cc)
    return (int(v.x) << 16) | int(v.y)


def _dm(proc: int = 0):
    return ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if proc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if proc == 0 else ttnn.NOC.RISCV_1_default,
    )


def chips(mesh) -> int:
    return int(mesh.shape[0]) * int(mesh.shape[1])


# ==================================================================================================== tables
class DeviceTable:
    """One engram layer's packed bfp8 table on the mesh: chip q holds rows [q nc, (q + 1) nc) as int32 pages of 64
    rows ([pages, 4352] int32 ROW_MAJOR DRAM interleaved, a page = 17408 B), uploaded once (``load``)."""

    def __init__(self, mesh, num_rows: int, layer: int):
        self.mesh, self.layer, self.num_rows = mesh, int(layer), int(num_rows)
        n = chips(mesh)
        self.nc = -(-self.num_rows // n)
        self.pages = -(-self.nc // ROWS_PER_PAGE)
        self.tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.pages, PAGE_WORDS]), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        self.load_s = None

    def rows_of(self, q: int) -> tuple[int, int]:
        lo = q * self.nc
        return lo, min(lo + self.nc, self.num_rows)

    def load(self, source, threads: int = 32) -> float:
        """Every chip's rows from ``source(lo, hi, out)`` (packed rows [hi - lo, 272] uint8 written into ``out``),
        uploaded a block of BLOCK_ROWS rows a chip at a time -> seconds."""
        t0 = time.time()
        n = chips(self.mesh)
        R, C = tuple(self.mesh.shape)
        block_pages = BLOCK_ROWS // ROWS_PER_PAGE
        host = torch.zeros(n, BLOCK_ROWS * ROW, dtype=torch.uint8)
        copier = _Copy(self.mesh)
        for b in range(-(-self.nc // BLOCK_ROWS)):
            pages = min(block_pages, self.pages - b * block_pages)
            shards = []
            for q in range(n):
                lo, hi = self.rows_of(q)
                a0 = lo + b * BLOCK_ROWS
                a1 = min(a0 + pages * ROWS_PER_PAGE, hi)
                buf = host[q, : pages * PAGE]
                if a1 > a0:
                    source(a0, a1, buf[: (a1 - a0) * ROW].view(a1 - a0, ROW).numpy())
                if a1 - a0 < pages * ROWS_PER_PAGE:
                    buf[max(0, a1 - a0) * ROW :].zero_()
                shards.append(ttnn.from_torch(buf.view(torch.int32).view(1, 1, pages, PAGE_WORDS), dtype=ttnn.int32))
            tmp = ttnn.to_device(ttnn.from_host_shards(shards, self.mesh.shape), self.mesh, ttnn.DRAM_MEMORY_CONFIG)
            copier(tmp, self.tensor, pages, b * block_pages)
            ttnn.deallocate(tmp)
        ttnn.synchronize_device(self.mesh)
        self.load_s = time.time() - t0
        return self.load_s


class _Copy:
    """Pages [0, n) of a [n, 4352] int32 tensor -> pages [p0, p0 + n) of the table (every chip its own)."""

    CORES = 16

    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, src, dst, n: int, p0: int):
        cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(self.CORES)]
        ct = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        for i, cc in enumerate(cores):
            rt[cc.x][cc.y] = [
                int(src.buffer_address()),
                int(dst.buffer_address()),
                n * i // len(cores),
                n * (i + 1) // len(cores),
                int(p0),
            ]
        crs = _crs(cores)
        prog = ttnn.ProgramDescriptor()
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_defines(_COPY),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=_dm(0),
            )
        ]
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=PAGE + 64,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, PAGE + 64)],
            )
        ]
        md = ttnn.MeshProgramDescriptor()
        R, C = tuple(self.mesh.shape)
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
        ttnn.generic_op([src, dst], md)


def checkpoint_source(ck, layer: int, threads: int = 32):
    """source(lo, hi, out) for DeviceTable.load: rows [lo, hi) of the layer's checkpoint table packed as bfloat8_b rows
    (tt/engram_bfp8.pack_into: the resident build's packing; pread, the file pages dropped after)."""
    from .engram import _drop_cache, _names
    from .engram_bfp8 import pack_into

    wname, sname = _names(layer)
    _, (_, d), woff, _ = ck.info(wname)
    _, (_, sd), soff, _ = ck.info(sname)
    fw = os.open(os.path.join(ck.path, ck.weight_map[wname]), os.O_RDONLY)
    fs = os.open(os.path.join(ck.path, ck.weight_map[sname]), os.O_RDONLY)

    def source(lo, hi, out):
        m = hi - lo
        w, s = torch.empty(m, d, dtype=torch.uint8), torch.empty(m, sd, dtype=torch.uint8)
        assert os.preadv(fw, [w.numpy()], woff + lo * d) == m * d
        assert os.preadv(fs, [s.numpy()], soff + lo * sd) == m * sd
        _drop_cache(fw, woff + lo * d, m * d)
        _drop_cache(fs, soff + lo * sd, m * sd)
        pack_into(w, s, out, threads)

    source.close = lambda: (os.close(fw), os.close(fs))
    return source


def resident_source(packed):
    """source(lo, hi, out) from a resident packed table [N, 272] (numpy)."""

    def source(lo, hi, out):
        out[:] = packed[lo:hi]

    source.close = lambda: None
    return source


def _barrett(p: int) -> tuple[int, int]:
    m = ((1 << 64) - 1) // int(p)
    return m & 0xFFFFFFFF, m >> 32


# ==================================================================================================== lookup
class EngramLookup:
    """The fused device engram lookup of every engram layer of a model (module docstring). ``__call__(ids_in, hist,
    start, valid)`` -> {layer: rows [1, 1, Q, 6144] bfloat8_b TILE per chip (the chip's own tokens; persistent)} and the
    natural-order ids [1, W] uint32 per chip (mesh row r: chunk offsets [r W, (r + 1) W), pads clamped) for the
    embedding."""

    def __init__(self, mesh, hasher, tables: dict, chunk: int, links: int | None = None):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan

        from .window_send import _torus, segments

        self.mesh, self.hasher, self.chunk = mesh, hasher, int(chunk)
        self.layers = sorted(tables)
        assert 1 <= len(self.layers) <= 2, self.layers
        self.tables = tables
        self.R, self.C = tuple(mesh.shape)
        self.n = self.R * self.C
        self.W = self.chunk // self.R
        self.Q = self.chunk // self.n
        assert self.Q % 32 == 0, (self.chunk, self.n)
        self.heads = int(hasher.n_heads)
        assert hasher.max_ngram_size == 4 and hasher.tail_len == 3, "the kernel's hash is for 4-grams"
        self.slots = 3 * self.heads
        self.width = self.slots * 256
        self.vocab = int(hasher.token_map.shape[0])
        dram = ttnn.DRAM_MEMORY_CONFIG
        rm = ttnn.ROW_MAJOR_LAYOUT
        # token map [vocab] -> [ceil(vocab / 16), 16] int32, replicated
        tm = hasher.token_map.to(torch.int32)
        pad = (-tm.numel()) % TMAP_PAGE
        tm = torch.cat([tm, torch.zeros(pad, dtype=torch.int32)]).view(1, 1, -1, TMAP_PAGE)
        self.tmap = ttnn.from_torch(tm, dtype=ttnn.int32, layout=rm, device=mesh, memory_config=dram)
        alloc = lambda shape, dt, lay=rm: ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, lay, mesh, dram)  # noqa
        self.ids_all = alloc([1, 1, self.R, self.W], ttnn.uint32)
        # the inbox: a region per source chip: its meta (count + an index word a row) then its row windows, a page = a
        # full packet of RPP windows (contiguous within a page; interleaved pages are not)
        L = len(self.layers)
        self.rpp = max(1, min(16, int(ttnn.get_tt_fabric_max_payload_size_bytes()) // STAGE))
        self.pkt = self.rpp * STAGE
        nrows = L * self.Q * self.slots
        self.meta_p = -(-((1 + nrows) * 4) // self.pkt)
        self.region = self.meta_p + -(-nrows // self.rpp)
        self.inbox = alloc([1, 1, self.n * self.region, self.pkt // 4], ttnn.int32)
        self.ptab = alloc([1, 1, nrows // 32, 32], ttnn.uint32)  # the hub's pointer table (convert reads it)
        self.out = {i: alloc([1, 1, self.Q, self.width], ttnn.bfloat8_b, ttnn.TILE_LAYOUT) for i in self.layers}
        self.emb_ids = alloc([1, self.W], ttnn.uint32)
        self.status = alloc([1, 1, 1, 16], ttnn.uint32)
        self.no_tail = new_tail(mesh)
        hub = ttnn.CoreCoord(0, 0)
        hcrs = _crs([hub])
        self.ids_sem = ttnn.create_global_semaphore(mesh, hcrs, 0, ttnn.BufferType.L1_SMALL)
        self.rows_sem = ttnn.create_global_semaphore(mesh, hcrs, 0, ttnn.BufferType.L1_SMALL)
        self.torus = _torus(mesh)
        self.links = int(settings.get("ENGRAM_DEVICE_LINKS") if links is None else links)  # (op tests: a sweep)
        self.ports, _, self.wraps = mesh_plan(mesh, self.links, self.torus)
        self.ids_seg, _ = segments(self.W * 4, int(ttnn.get_tt_fabric_max_payload_size_bytes()))
        self._pp = None
        self.cores = None

    # ------------------------------------------------------------------ host plan
    def owned(self, q: int, layer: int) -> list:
        """(slot, prime, barrett lo, hi, offset) of the (n, head) sub-tables whose rows overlap chip q's range."""
        h = self.hasher
        li = h.layer_ids.index(layer)
        lo, hi = self.tables[layer].rows_of(q)
        out = []
        for slot in range(self.slots):
            p = int(h.primes[li].reshape(-1)[slot])
            off = int(h.offsets[li][slot])
            if off < hi and off + p > lo:
                out.append((slot, p, *_barrett(p), off))
        return out

    def _node(self, ch) -> int:
        nd = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))
        return (int(nd.mesh_id) << 16) | int(nd.chip_id)

    def _io(self, ids_in, hist, tail):
        return [ids_in, hist, tail] + [self.out[i] for i in self.layers] + [self.emb_ids, self.status]

    def _build(self, io):
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        ids_in, hist, tail, emb, status = io[0], io[1], io[2], io[-2], io[-1]
        out0, out1 = io[3], io[-3]
        mesh, R, C, n = self.mesh, self.R, self.C, self.n
        hub = ttnn.CoreCoord(0, 0)
        send_cores = [ttnn.CoreCoord(1 + i, 0) for i in range(1 + 4 * self.links)]
        hash_cores = [ttnn.CoreCoord(i % 8, 1 + i // 8) for i in range(HASH_CORES)]
        conv_cores = [ttnn.CoreCoord(i % 8, 1 + HASH_CORES // 8 + i // 8) for i in range(CONVERT_CORES)]
        self.cores = 1 + len(hash_cores) + len(conv_cores) + len(send_cores)
        hash_xy = [_xy(mesh, c) for c in hash_cores]
        conv_xy = [_xy(mesh, c) for c in conv_cores]
        hub_xy = _xy(mesh, hub)
        nodes = [self._node(divmod(q, C)) for q in range(n)]
        mult = {}
        for layer in self.layers:
            li = self.hasher.layer_ids.index(layer)
            words = []
            for s in range(4):
                v = int(self.hasher.multipliers[li][s]) & ((1 << 64) - 1)
                words += [v & 0xFFFFFFFF, v >> 32]
            mult[layer] = words
        L = len(self.layers)
        tabs = [self.tables[i].tensor for i in self.layers] + [self.tables[self.layers[0]].tensor] * (2 - L)
        outs = [out0, out1 if L > 1 else out0]
        ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        max_items = (self.chunk // HASH_CORES) * self.slots * L
        items_bytes = 16 + 8 * max_items
        md = ttnn.MeshProgramDescriptor()
        for r in range(R):
            for c in range(C):
                q = r * C + c
                ch = (r, c)
                # senders: one per first hop direction of the routes leaving this chip, then the local one
                dirs = []
                dest_sender = []
                per_dir = {}
                for d in range(n):
                    dc = divmod(d, C)
                    if dc == ch:
                        key = "local"
                    else:
                        hops = route(ch, dc, R, C, self.wraps[0], self.wraps[1], 0)
                        k0 = per_dir.get(hops[0][1], 0)
                        per_dir[hops[0][1]] = k0 + 1
                        key = (hops[0][1], k0 % self.links)  # (the direction's links take its destinations in turn)
                    if key not in dirs:
                        dirs.append(key)
                    dest_sender.append(dirs.index(key))
                assert len(dirs) <= len(send_cores), dirs
                peers = {k: [] for k in dirs}
                for r2 in range(R):
                    if r2 != r:
                        d = r2 * C + c
                        peers[dirs[dest_sender[d]]].append(d)
                prog = ttnn.ProgramDescriptor()
                used = [hub] + hash_cores + conv_cores + send_cores[: len(dirs)]
                all_crs = _crs(used)
                prog.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in (0, 1)]
                cbs = [
                    ttnn.CBDescriptor(
                        total_size=items_bytes,
                        core_ranges=all_crs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, items_bytes)],
                    )
                ]
                scratch = {
                    "hub": (
                        self.chunk * 4 + 64 + self.W * 4 + 128 + L * self.slots * self.Q * 4 + self.pkt + 64,
                        [hub],
                    ),
                    "hash": (
                        self.chunk * 4 + 2 * (self.chunk // HASH_CORES + 3) * 4 + 6 * 64 + 64 * 64 + 256,
                        hash_cores,
                    ),
                    "conv": (32 * STAGE + 8 * 1088 + 128 + 64, conv_cores),
                    "send": (
                        2 * self.rpp * STAGE + 256 * 8 + 16 * 160 + (1 + L * self.Q * self.slots) * 4 + 128,
                        send_cores[: len(dirs)],
                    ),
                }
                for size, cs in scratch.values():
                    cbs.append(
                        ttnn.CBDescriptor(
                            total_size=size,
                            core_ranges=_crs(cs),
                            format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint8, size)],
                        )
                    )
                prog.cbs = cbs
                kernels = []
                # hub (kernel 0)
                rt = ttnn.RuntimeArgs()
                rt[hub.x][hub.y] = (
                    [
                        0,
                        0,
                        int(ids_in.buffer_address()),
                        int(self.ids_all.buffer_address()),
                        int(emb.buffer_address()),
                        int(status.buffer_address()),
                        int(tail.buffer_address()),
                        int(self.inbox.buffer_address()),
                        int(self.ptab.buffer_address()),
                        r,
                        q,
                        int(ttnn.get_global_semaphore_address(self.ids_sem)),
                        int(ttnn.get_global_semaphore_address(self.rows_sem)),
                        0,
                        len(hash_xy),
                    ]
                    + hash_xy
                    + [len(conv_xy)]
                    + conv_xy
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_defines(_HUB),
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=_crs([hub]),
                        compile_time_args=[
                            self.W,
                            R,
                            self.chunk,
                            self.Q,
                            L,
                            self.slots,
                            self.vocab,
                            n,
                            self.pkt,
                            self.meta_p,
                            self.region,
                        ]
                        + ta(ids_in)
                        + ta(self.ids_all)
                        + ta(emb)
                        + ta(status)
                        + ta(tail)
                        + ta(self.inbox)
                        + ta(self.ptab),
                        runtime_args=rt,
                        config=_dm(0),
                    )
                )
                # hash cores (kernel 1)
                send_xy = [_xy(mesh, send_cores[k]) for k in range(len(dirs))]
                lay = []
                for layer in self.layers:
                    own = self.owned(q, layer)
                    lay += mult[layer] + [self.tables[layer].rows_of(q)[0], self.tables[layer].nc, len(own)]
                    for e in own:
                        lay += list(e)
                rt = ttnn.RuntimeArgs()
                step = self.chunk // HASH_CORES
                for k, cc in enumerate(hash_cores):
                    rt[cc.x][cc.y] = (
                        [
                            0,
                            0,
                            int(self.ids_all.buffer_address()),
                            int(hist.buffer_address()),
                            int(self.tmap.buffer_address()),
                            int(tail.buffer_address()),
                            k * step,
                            (k + 1) * step,
                            0,
                            1,
                            len(send_xy),
                        ]
                        + send_xy
                        + [n]
                        + dest_sender
                        + lay
                    )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_defines(_HASH),
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=_crs(hash_cores),
                        compile_time_args=[
                            self.W,
                            R,
                            self.chunk,
                            self.Q,
                            L,
                            self.heads,
                            self.vocab,
                            int(self.hasher.pad_id),
                            max_items,
                        ]
                        + ta(self.ids_all)
                        + ta(hist)
                        + ta(self.tmap)
                        + ta(tail),
                        runtime_args=rt,
                        config=_dm(0),
                    )
                )
                # convert cores (kernel 2)
                units = L * (self.Q // 32) * self.slots
                rt = ttnn.RuntimeArgs()
                for k, cc in enumerate(conv_cores):
                    rt[cc.x][cc.y] = [
                        0,
                        0,
                        int(self.inbox.buffer_address()),
                        hub_xy,
                        int(self.ptab.buffer_address()),
                        int(outs[0].buffer_address()),
                        int(outs[1].buffer_address()),
                        q,
                        0,
                        units * k // len(conv_cores),
                        units * (k + 1) // len(conv_cores),
                    ]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_defines(_CONV),
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=_crs(conv_cores),
                        compile_time_args=[self.Q, self.slots, L, self.pkt, self.meta_p, self.region, n]
                        + ta(self.inbox)
                        + ta(self.ptab)
                        + ta(outs[0])
                        + ta(outs[1]),
                        runtime_args=rt,
                        config=_dm(0),
                    )
                )
                # senders (kernels 3..)
                for k, key in enumerate(dirs):
                    cc = send_cores[k]
                    fabric = key != "local"
                    args = [
                        int(ids_in.buffer_address()),
                        int(self.ids_all.buffer_address()),
                        int(tabs[0].buffer_address()),
                        int(tabs[1].buffer_address()),
                        int(self.inbox.buffer_address()),
                        q,
                        k,
                        r,
                        hub_xy,
                        int(ttnn.get_global_semaphore_address(self.ids_sem)),
                        int(ttnn.get_global_semaphore_address(self.rows_sem)),
                        1,
                        len(hash_xy),
                    ]
                    mine = [d for d in range(n) if dest_sender[d] == k]
                    args += hash_xy + [n] + nodes + [len(peers[key])] + peers[key] + [len(mine)] + mine
                    if fabric:
                        port = self.ports[ch][key[0]]
                        args += list(
                            ttnn.setup_fabric_connection(
                                mesh,
                                mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch)),
                                mesh.get_fabric_node_id(ttnn.MeshCoordinate(*port["peer"])),
                                port["links"][key[1]],
                                prog,
                                cc,
                            )
                        )
                    rt = ttnn.RuntimeArgs()
                    rt[cc.x][cc.y] = args
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_defines(_SEND),
                            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                            core_ranges=_crs([cc]),
                            compile_time_args=[
                                self.W,
                                L,
                                int(fabric),
                                self.ids_seg,
                                self.rpp,
                                self.pkt,
                                self.meta_p,
                                self.region,
                            ]
                            + ta(ids_in)
                            + ta(self.ids_all)
                            + ta(tabs[0])
                            + ta(tabs[1])
                            + ta(self.inbox),
                            runtime_args=rt,
                            config=_dm(0),
                        )
                    )
                prog.kernels = kernels
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        return md

    def __call__(self, ids_in, hist, start: int, valid: int, tail=None):
        L = self.layers
        io = self._io(ids_in, hist, tail if tail is not None else self.no_tail)
        if self._pp is None:
            md = self._build(io)
            self._pp = ttnn.bringup.prepare_program(
                md, io, value_slots=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
            )
        s, v = int(start), int(valid)
        ttnn.bringup.run_prepared(self._pp, io, [s, v, s, v, s, v])
        return {i: self.out[i] for i in L}, self.emb_ids

    def read_status(self) -> list:
        return [ttnn.to_torch(t).reshape(-1)[:8].tolist() for t in ttnn.get_device_tensors(self.status)]


def new_history(mesh, max_seq: int):
    """A slot's device id history [ceil(max_seq / 32), 32] uint32 per chip (replicated)."""
    pages = -(-int(max_seq) // HIST_PAGE)
    return ttnn.from_torch(
        torch.zeros(1, 1, pages, HIST_PAGE, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def new_tail(mesh, comp=None):
    """A slot's explicit tail [1, 1, 1, 16] uint32 per chip: [flag, c0, c1, c2] (flag 1: the next lookup takes these
    compressed ids (DEAD = 0xFFFFFFFF) for positions start - 3 .. start - 1 instead of the id history; the lookup
    clears it)."""
    v = torch.zeros(16, dtype=torch.int64)
    if comp is not None:
        v[0] = 1
        v[1:4] = comp.reshape(-1).long() & 0xFFFFFFFF
    return ttnn.from_torch(
        v.to(torch.uint32).view(1, 1, 1, 16),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def engine_ids(mesh, tokens: torch.Tensor, start: int, chunk: int):
    """Host natural-order ids [chunk] (pads past valid) -> the engine's device input for a chunk at ``start``
    (uint32 [SP, 1, W] in the server's order, sharded over axis 0, replicated over axis 1; adapter.make_chunk_input)."""
    from .attention import WindowGeometry

    g = WindowGeometry(mesh, chunk)
    nat = tokens.reshape(-1).long() & 0xFFFFFFFF
    srv = nat[g.server_rows(start).reshape(-1)].reshape(g.sp, 1, g.w)
    return ttnn.from_torch(
        srv.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )


def table_bytes_a_chip(mesh, num_rows: int) -> int:
    nc = -(-int(num_rows) // chips(mesh))
    return -(-nc // ROWS_PER_PAGE) * PAGE


def fits(mesh, cfg) -> bool:
    """Every engram table of the model on this mesh within ENGRAM_DEVICE_MAX_GB a chip (8x4: 6.6 GB; 4x4 13 GB and
    the smaller meshes do not fit: ENGRAM_DEVICE=on runs the host path there)."""
    total = sum(table_bytes_a_chip(mesh, n) for n in cfg.raw["engram_num_embeddings"])
    return total <= float(settings.get("ENGRAM_DEVICE_MAX_GB")) * 1e9


TABLES = {}  # (id(mesh), checkpoint, layer) -> DeviceTable: one upload a process


def load_table(mesh, ck, layer: int, num_rows: int, cache_dir=None) -> DeviceTable:
    """The layer's DeviceTable, uploaded once a process. In the bring-up's resident runner it goes through the device
    weight registry (testing/weights.load_tensor) under a marker file of the table's byte size (sparse, no data), so
    a model rebuilt in the same process keeps it; elsewhere a module-level cache."""
    key = (id(mesh), str(ck.path), int(layer))
    hit = TABLES.get(key)
    if hit is not None and hit.tensor.is_allocated():
        return hit

    def build(path=None, device=None):
        from .engram import _RESIDENT

        t = DeviceTable(mesh, num_rows, layer)
        packed = _RESIDENT.get((str(ck.path), "bfp8", layer))
        src = resident_source(packed.numpy()) if packed is not None else checkpoint_source(ck, layer)
        try:
            t.load(src)
        finally:
            src.close()
        return t

    table = None
    try:
        from models.demos.common.bringup.testing import weights

        if weights.active() and cache_dir:
            # (a copy this process already holds outside a model build, e.g. an op-level test's: kept, not reloaded)
            for pool in (weights._BUILT, weights._AVAIL):
                for k, ts in pool.items():
                    if (
                        k[0] == "file"
                        and k[1].endswith(f"/L{layer}_{num_rows}_{chips(mesh)}chips_v1.marker")
                        and k[-1] == id(mesh)
                    ):
                        alive = [
                            x for x in ts if hasattr(x, "rows_of") and x.tensor.is_allocated()
                        ]  # (any module copy)
                        if alive and pool is weights._BUILT:
                            TABLES[key] = alive[0]
                            return alive[0]
            d = os.path.join(str(cache_dir), "engram_device")
            os.makedirs(d, exist_ok=True)
            n = chips(mesh)
            size = table_bytes_a_chip(mesh, num_rows) * n
            marker = os.path.join(d, f"L{layer}_{num_rows}_{n}chips_v1.marker")
            if not os.path.exists(marker) or os.path.getsize(marker) != size:
                with open(marker, "wb") as f:
                    f.truncate(size)  # (sparse: the registry's kept-bytes count reads the size)
            table = weights.load_tensor(marker, mesh, load=build)
    except ImportError:
        table = None
    if table is None:
        table = build()
    TABLES[key] = table
    return table


def _is_allocated(self) -> bool:
    return self.tensor.is_allocated()


DeviceTable.is_allocated = _is_allocated
