// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Indexer phase: the running top-512 filter of one RISC over rows [ra, rb) of its score CB (ixp_common.hpp).
// Per query: a min-heap of the 512 largest (key, pool) pairs seen so far (thr = its root once full; a key must exceed
// it). At the end the 512 pool ids per query (fewer
// visible: 0xFFFFFFFF padding) -> row q0 + r of the selection.
#pragma once

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/ix/ixp_common.hpp"

namespace ixp {

struct Cand {
    uint32_t key;
    uint32_t pool;
};

// the k largest keys of a[0, n) moved to a[0, k) (in place, any order); n > k
__attribute__((optimize("O2"))) inline void select_top(Cand* a, uint32_t n, uint32_t k) {
    uint32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const uint32_t x = a[lo].key, y = a[mid].key, z = a[hi].key;
        const uint32_t piv = (x > y) ? ((y > z) ? y : ((x > z) ? z : x)) : ((x > z) ? x : ((y > z) ? z : y));
        uint32_t i = lo, j = hi;
        while (i <= j) {
            while (a[i].key > piv) {
                ++i;
            }
            while (a[j].key < piv) {
                --j;
            }
            if (i <= j) {
                const Cand t = a[i];
                a[i] = a[j];
                a[j] = t;
                ++i;
                if (j == 0) {
                    break;
                }
                --j;
            }
        }
        if (k - 1 <= j) {
            hi = j;
        } else if (k - 1 >= i) {
            lo = i;
        } else {
            break;
        }
    }
}

// IXP_CS_TENSOR (the served layer: one trace replayed for every chunk): CS = chunk start + the chip's first row, read
// from two one-element int32 DRAM tensors (expand_rows' start / base) into 64-B aligned scratch slots (the DRAM read
// alignment)
template <typename A0, typename A1>
inline uint32_t read_cs(const A0& start, const A1& base, uint32_t scratch) {
    noc_async_read(start.get_noc_addr(0), scratch, 4);
    noc_async_read(base.get_noc_addr(0), scratch + 64, 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    return v[0] + v[16];
}

template <uint32_t CAP>
struct Filter {
    Cand* cand;  // (rb - ra) x CAP
    uint32_t ra, rb;
    uint32_t n[32], thr[32], vis[32];

    void init(uint32_t cand_addr, uint32_t ra_, uint32_t rb_, uint32_t cs, uint32_t q0, uint32_t nq, uint32_t pools) {
        cand = reinterpret_cast<Cand*>(cand_addr);
        ra = ra_;
        rb = rb_;
        for (uint32_t r = ra; r < rb; ++r) {
            n[r - ra] = 0;
            thr[r - ra] = 0;  // every finite score's key is >= 1
            const uint32_t v = (cs + q0 + r + 1) / 4;
            vis[r - ra] = r < nq ? (v < pools ? v : pools) : 0;
        }
    }

    // the KC score tiles of block b at L1 address sb (fp32, row r of tile j = query r's scores of key tile b KC + j)
    template <uint32_t KC>
    __attribute__((optimize("O2"))) void block(uint32_t sb, uint32_t b) {
        asm volatile("" ::: "memory");  // the scores landed (after the caller's cb_wait_front)
#if defined(IXP_DIAG) && (IXP_DIAG & 1)
        return;  // diagnostic: no scan
#endif
        for (uint32_t j = 0; j < KC; ++j) {
            const uint32_t p0 = (b * KC + j) * 32;
            const uint32_t* t = reinterpret_cast<const uint32_t*>(sb + j * T32);
            const uint32_t* mx = reinterpret_cast<const uint32_t*>(sb + (KC + j) * T32);  // v8 row maxima (column 0)
            for (uint32_t r = ra; r < rb; ++r) {
                const uint32_t i = r - ra;
                if (p0 >= vis[i]) {
                    continue;
                }
                if (n[i] == TOPK) {  // the heap is full: skip a row-tile whose max cannot beat its root
                    const uint32_t mk = okey(mx[tile_off(r, 0)]);
                    if (mk <= 0xFFFFFFFFu - MAX_MARGIN && mk + MAX_MARGIN <= thr[i]) {
                        continue;
                    }
                }
                const uint32_t lim = vis[i] - p0 < 32 ? vis[i] - p0 : 32;
                // the row's two 16-score face halves, 4 scores per step (branchless keys; an insert is rare)
                for (uint32_t hf = 0; hf < 2; ++hf) {
                    const uint32_t c0 = 16 * hf;
                    if (c0 >= lim) {
                        break;
                    }
                    const uint32_t cn = lim - c0 < 16 ? lim - c0 : 16;
                    const uint32_t* h = t + tile_off(r, c0);
                    uint32_t c = 0;
                    for (; c + 4 <= cn; c += 4) {
                        const uint32_t k0 = okey(h[c]), k1 = okey(h[c + 1]), k2 = okey(h[c + 2]), k3 = okey(h[c + 3]);
                        const uint32_t th = thr[i];
                        if ((k0 > th) | (k1 > th) | (k2 > th) | (k3 > th)) {
                            insert(i, k0, p0 + c0 + c);
                            insert(i, k1, p0 + c0 + c + 1);
                            insert(i, k2, p0 + c0 + c + 2);
                            insert(i, k3, p0 + c0 + c + 3);
                        }
                    }
                    for (; c < cn; ++c) {
                        insert(i, okey(h[c]), p0 + c0 + c);
                    }
                }
            }
        }
    }

    // key -> row i's top-512 kept as a MIN-HEAP of (key, pool) (v6): below 512 entries a push (sift up), then a key
    // above the root replaces it (sift down, 9 levels); thr = the root once full. O(log 512) per accepted score and no
    // bursts (v5's quickselect on overflow stalled the member ~75 us, and the lockstep multicast stalled every member)
    __attribute__((optimize("O2"))) inline void insert(uint32_t i, uint32_t key, uint32_t pool) {
        if (key <= thr[i]) {
            return;
        }
        Cand* h = cand + i * CAP;
        uint32_t m = n[i];
        if (m < TOPK) {  // push + sift up
            uint32_t c = m;
            while (c > 0) {
                const uint32_t p = (c - 1) >> 1;
                if (h[p].key <= key) {
                    break;
                }
                h[c] = h[p];
                c = p;
            }
            h[c].key = key;
            h[c].pool = pool;
            n[i] = ++m;
            if (m == TOPK) {
                thr[i] = h[0].key;
            }
            return;
        }
        uint32_t c = 0;  // replace the root + sift down
        for (;;) {
            uint32_t l = 2 * c + 1;
            if (l >= TOPK) {
                break;
            }
            const uint32_t r = l + 1;
            if (r < TOPK && h[r].key < h[l].key) {
                l = r;
            }
            if (h[l].key >= key) {
                break;
            }
            h[c] = h[l];
            c = l;
        }
        h[c].key = key;
        h[c].pool = pool;
        thr[i] = h[0].key;
    }

    template <typename Acc>
    void finish(const Acc& ss, uint32_t q0, uint32_t nq) {
        for (uint32_t r = ra; r < rb && r < nq; ++r) {
            const uint32_t i = r - ra;
            Cand* cr = cand + i * CAP;  // the heap: n[i] <= 512 entries, any order is the set
            uint32_t* row = reinterpret_cast<uint32_t*>(cr + TOPK);  // CAP >= 2 TOPK: free behind the kept 512
            for (uint32_t q = 0; q < TOPK; ++q) {
                row[q] = q < n[i] ? cr[q].pool : 0xFFFFFFFFu;
            }
            noc_async_write(reinterpret_cast<uint32_t>(row), ss.get_noc_addr(q0 + r), TOPK * 4);
            noc_async_write_barrier();
        }
    }
};

}  // namespace ixp
