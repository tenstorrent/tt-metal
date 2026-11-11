// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// T16 (DSV4.1-Flash prefill): K-row cache across the consecutive query tokens of one core.
//
// The op is bound by the random-row gather of the K rows (window 128 + top-512 selected = 640 rows of 512 B per query
// token). Consecutive query tokens of a core select almost the same rows (the 128-row sliding windows overlap in 127
// rows, the top-512 selections overlap strongly), so when the K chunk is the WHOLE index row (k_chunk == topk, python:
// DSV41_PFA_SP_KC=640) cb_k_rm (one chunk = topk rows, single buffered: the same L1 addresses every token) is kept as a
// cache: for token t+1 only the rows that token t did not hold are gathered, into the slots of the rows token t held
// and token t+1 does not need. Softmax attention is invariant to the order of the keys, so the slot order does not
// matter (the summation order changes: not bit-identical, PCC ~1).
//
// The bookkeeping lives in the 24 padded head rows of cb_q_rm: the DSV4 prefill passes 8 real heads padded to 32
// (rows 8..31 are zero and their outputs are discarded). This kernel does not read the padded rows (saves 24 DRAM reads
// per token); the compute kernel tilizes whatever is in them (bookkeeping bytes: garbage in the padded heads' rows
// only, rows are independent in the matmuls / reduces / softmax). Only used when H == 32.
#pragma once

#include <stdint.h>

namespace sparse_sdpa_dedup {

constexpr uint32_t Q_REAL_HEADS = 8;
constexpr uint32_t EMPTY = 0xFFFFFFFFu;  // free hash bucket / invalid slot id
constexpr uint32_t TOMB = 0xFFFFFFFEu;   // deleted hash bucket
constexpr uint32_t HASH_BITS = 11;       // 2048 buckets, <= 1024 ids
constexpr uint32_t HASH_BUCKETS = 1u << HASH_BITS;
constexpr uint32_t HASH_MASK = HASH_BUCKETS - 1;
constexpr uint32_t REBUILD_MISSES = 160;  // above this many misses a token rebuilds the hash from the slot table (~13 us) instead of erase + insert per miss
constexpr uint32_t LIST_MARK = 0xFFFFFFFEu;  // gather_request::RUN value: list mode (BASE = L1 address of the miss
                                             // list, SPLIT = entries for the writer)

// Persistent state of one core: slot table S (slot -> pool row id), open-addressing hash table (key = id, value = slot)
// kept in step with S (deleted buckets are tombstones, the table is rebuilt when they pile up), per-token scratch
// (used-slot bitmap, miss list).
template <uint32_t NK>
struct Scratch {
    volatile uint32_t* S;
    volatile uint32_t* hk;
    volatile uint16_t* hv;
    volatile uint32_t* miss;
    volatile uint32_t* used;
    uint32_t ntomb = 0;
    static constexpr uint32_t WORDS = NK + HASH_BUCKETS + HASH_BUCKETS / 2 + NK + (NK + 31) / 32;
    FORCE_INLINE explicit Scratch(uint32_t base) {
        S = reinterpret_cast<volatile uint32_t*>(base);
        hk = S + NK;
        hv = reinterpret_cast<volatile uint16_t*>(hk + HASH_BUCKETS);
        miss = reinterpret_cast<volatile uint32_t*>(hv + HASH_BUCKETS);
        used = miss + NK;
    }
};

FORCE_INLINE uint32_t hash_id(uint32_t id) { return (id * 2654435761u) >> (32 - HASH_BITS); }

template <uint32_t NK>
FORCE_INLINE void hash_insert(Scratch<NK>& sc, uint32_t id, uint32_t slot) {
    uint32_t h = hash_id(id);
    uint32_t k;
    while ((k = sc.hk[h]) < TOMB) {
        h = (h + 1) & HASH_MASK;
    }
    if (k == TOMB) {
        --sc.ntomb;
    }
    sc.hk[h] = id;
    sc.hv[h] = static_cast<uint16_t>(slot);
}

template <uint32_t NK>
FORCE_INLINE void hash_erase(Scratch<NK>& sc, uint32_t id) {
    uint32_t h = hash_id(id);
    uint32_t k;
    while ((k = sc.hk[h]) != EMPTY) {
        if (k == id) {
            sc.hk[h] = TOMB;
            ++sc.ntomb;
            return;
        }
        h = (h + 1) & HASH_MASK;
    }
}

template <uint32_t NK>
FORCE_INLINE void hash_rebuild(Scratch<NK>& sc) {
    for (uint32_t i = 0; i < HASH_BUCKETS; ++i) {
        sc.hk[i] = EMPTY;
    }
    sc.ntomb = 0;
    for (uint32_t s = 0; s < NK; ++s) {
        hash_insert(sc, sc.S[s], s);
    }
}

// Plan the gathers of one token whose NK ids are idx[0..NK). Returns the number of miss-list entries (entry = (j << 16)
// | slot: gather page idx[j] into slot); updates the slot table S and the hash. Cold (cache invalid or inconsistent):
// every row goes to slot j.
template <uint32_t NK>
FORCE_INLINE uint32_t plan(Scratch<NK>& sc, volatile tt_l1_ptr uint32_t* idx, bool& s_valid) {
    uint32_t nm = 0;
    bool cold = !s_valid;
    if (!cold) {
        for (uint32_t i = 0; i < (NK + 31) / 32; ++i) {
            sc.used[i] = 0;
        }
        // 1. which new ids are cached (mark their slot used); the others go to the miss list (by j)
        for (uint32_t j = 0; j < NK; ++j) {
            const uint32_t id = idx[j];
            uint32_t h = hash_id(id);
            uint32_t k;
            while ((k = sc.hk[h]) != EMPTY) {
                if (k == id) {
                    break;
                }
                h = (h + 1) & HASH_MASK;
            }
            if (k == id) {
                const uint32_t slot = sc.hv[h];
                sc.used[slot >> 5] |= 1u << (slot & 31);
            } else {
                sc.miss[nm++] = j;
            }
        }
        // distinct cached slots hit must equal NK - nm (else the row has duplicate ids: rebuild)
        uint32_t nused = 0;
        for (uint32_t i = 0; i < (NK + 31) / 32; ++i) {
            nused += __builtin_popcount(sc.used[i]);
        }
        // 2. free slots (not used) take the missing rows. Many misses (low overlap): rebuilding the whole hash once is cheaper than an erase + insert per miss.
        const bool rebuild_all = nm > REBUILD_MISSES;
        uint32_t m = 0;
        for (uint32_t w = 0; w < (NK + 31) / 32 && m < nm && nused == NK - nm; ++w) {
            uint32_t f = ~sc.used[w];
            if (w == NK / 32) {
                f &= (1u << (NK % 32)) - 1u;  // (only when NK % 32 != 0)
            }
            while (f != 0 && m < nm) {
                const uint32_t b = __builtin_ctz(f);
                f &= f - 1;
                const uint32_t slot = (w << 5) + b;
                const uint32_t j = sc.miss[m];
                const uint32_t id = idx[j];
                if (!rebuild_all) {
                    const uint32_t old = sc.S[slot];
                    if (old != EMPTY) {
                        hash_erase(sc, old);
                    }
                    hash_insert(sc, id, slot);
                }
                sc.S[slot] = id;
                sc.miss[m] = (j << 16) | slot;
                ++m;
            }
        }
        if (m != nm || nused != NK - nm) {
            cold = true;  // duplicate ids in the row (fewer distinct rows than slots): rebuild
        } else if (rebuild_all || sc.ntomb > 700) {
            hash_rebuild(sc);
        }
    }
    if (cold) {
        nm = 0;
        for (uint32_t j = 0; j < NK; ++j) {
            sc.S[j] = idx[j];
            sc.miss[nm++] = (j << 16) | j;
        }
        hash_rebuild(sc);
    }
    s_valid = true;
    return nm;
}

// Gather list entries [lo, hi) of the miss list (trid ring of depth D, like sparse_sdpa::trid_ring_gather).
template <uint32_t D, typename Accessor>
FORCE_INLINE void gather_list(
    Noc& noc,
    const Accessor& kv,
    uint32_t dst_l1,
    volatile tt_l1_ptr uint32_t* idx,
    volatile uint32_t* list,
    uint32_t lo,
    uint32_t hi,
    uint32_t k_row_bytes,
    uint32_t page_offset) {
    const UnicastEndpoint local_l1;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t e = list[lo + i];
        noc.async_read(
            kv,
            local_l1,
            k_row_bytes,
            {.page_id = page_offset + idx[e >> 16]},
            {.addr = dst_l1 + (e & 0xFFFFu) * k_row_bytes});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);
}

}  // namespace sparse_sdpa_dedup
