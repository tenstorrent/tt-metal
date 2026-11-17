// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// WL_HA2A (lane design, LCfg.ha2a): h rows as a column all-to-all instead of column leaders + chains (needs WL_HCRED).
// Core (C, Y) owns h tile t = HALF C + Y of every row. Its BRISC copies the tile from CB 16 into its own ring slot,
// then its flag (one 16 B word a tile, KTD a slot, at WL_HA2A_F); each RISC passes two tiles of the row on - its own
// and its partner's (t' = HALF (1 - C) + Y, once landed) - to the rows of its column on its side (BRISC below on NOC0,
// NCRISC above on NOC1), and its own tile to the partner (column 0's BRISC east, column 1's NCRISC west): the payload
// and its flag word (the landed local flag word as the source: value q + 1, offset 0 like the target) from one command
// buffer, once the target has consumed row q - NS (its WL_HCRED word). At most two hops (to the partner, then down its
// column); a row is complete once all KTD flags read >= q + 1.
#pragma once
#include "api/dataflow/dataflow_api.h"

struct HFwd {
    uint32_t n = 0, k = 0, q = 0;
    uint8_t tile[24], tx[24], ty[24], word[24];
};

template <uint32_t HALF>
FORCE_INLINE void hfwd_init(HFwd& h, bool down, uint32_t C, uint32_t Y) {
    const uint32_t to = HALF * C + Y, tp = HALF * (1 - C) + Y;
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t t = i == 0 ? to : tp;
        for (uint32_t o = 0; o < HALF; ++o) {
            if (down ? o > Y : o < Y) {
                h.tile[h.n] = t, h.tx[h.n] = C, h.ty[h.n] = o, h.word[h.n] = o, ++h.n;
            }
        }
        if (i == 0 && (down ? C == 0 : C == 1)) {
            h.tile[h.n] = t, h.tx[h.n] = 1 - C, h.ty[h.n] = Y, h.word[h.n] = HALF, ++h.n;
        }
    }
}

// pass row h.q's tiles on (as far as they have landed and their targets have room); h.q advances once all have left
template <uint32_t NS, uint32_t KTD>
FORCE_INLINE bool hfwd_step(HFwd& h, uint32_t total, const uint32_t* cx, uint32_t ny0, uint32_t hring, uint32_t htile) {
    if (h.q >= total) {
        return false;
    }
    bool did = false;
    const uint32_t q = h.q, s = q % NS;
    invalidate_l1_cache();
    while (h.k < h.n) {
        const uint32_t t = h.tile[h.k], f = WL_HA2A_F + (s * KTD + t) * 16;
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(f) < q + 1) {
            return did;
        }
        if (q >= NS && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HCRED + h.word[h.k] * 16) < q - NS + 1) {
            return did;
        }
        const uint32_t a = hring + (s * KTD + t) * htile;
        const uint32_t x = cx[h.tx[h.k]], y = ny0 + h.ty[h.k];
        noc_async_write(a, get_noc_addr(x, y, a), htile);
        noc_async_write(f, get_noc_addr(x, y, f), 4);
        ++h.k;
        did = true;
    }
    noc_async_writes_flushed();  // (the row's slot is the source: out of L1 before the row counts as forwarded)
    h.k = 0;
    ++h.q;
    return true;
}
