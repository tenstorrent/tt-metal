// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7b MLA megakernel — the attention phase (flash-decoding over the paged latent cache, in-program):
// owner core (g, c) of G head groups x C row chunks takes heads [32 g, 32 g + 32) and, per token t, the row tiles
// jt = c, c + C, c + 2C, ... < n_t = p_t / 32 + 1 of that token's latent rows (p_t = cur_pos[t]; the dense-exact range
// D1: rows 0..p_t; the sparse path S1 swaps the identity for a row list), computes the partial (m, l, O) of its rows
//   S = q_lat . K^T * scale (K = V = the latent rows), rows > p_t masked; m = rowmax S; P = exp(S - m); l = rowsum P;
//   O = P @ V
// and the C partials of a group are combined in a binary tree (level L: core c with c % 2^(L+1) == 0 receives from
// c + 2^L) to o = O / l on core c = 0, which writes the group's [32 heads, 512] tile row of o_lat [1, T, H, 512].
// A partial is two CBs (fp32): the stats (0 = m, 1 = l; column 0) and O (16 tiles of 32 heads x 32 columns); a tree
// message writes both into the parent's receive CBs (CB_ARX, CB_ARXO).
#pragma once

#include <cstdint>

namespace mlaatt {
constexpr uint32_t PART_TILES = 18;
constexpr uint32_t STAT_TILES = 2;  // m, l
constexpr uint32_t DT = 16;         // latent tiles (512 / 32)
// row tiles per partial (tt/mk_mla.py ATT_NKMAX: CB_AK = NKMAX x DT pages, CB_AS / CB_AE / CB_AP = NKMAX pages). The
// compute reads these CBs by page index from the read pointer and the unpacker does NOT wrap at the CB's end, so every
// partial pads them to their capacity (pushes of unwritten pages): each partial starts at the CB base.
#if defined(MLA_ATT_NKMAX)  // rev 32 (tt/mk_mla.py att_nkmax, t > 4 dense): bigger batches, fewer O rescales
constexpr uint32_t NKMAX = MLA_ATT_NKMAX;
#else
constexpr uint32_t NKMAX = 3;
#endif
// CB_AINFO word 2 of a token's plan: the token's partial comes in batches of <= NKMAX tiles, each announced by its own
// plan page (bk, bm = the masked tile or 0xFFFF, last) -- the token lanes (MLA_ATT_LANES) and DSA's sparse tokens
// (0x5C)
constexpr uint32_t KIND_BATCHED = 0xB7;
// MLA_ATT_LANES (tt/mk_mla.py att_lanes, t > 4): the C owners of a head group form LANES token lanes of C / LANES
// chunk cores; lane l takes tokens l, l + LANES, ... with its own partial tree (compile-time C / LV = the lane's).
// With DSA (Stage 14 rev 40, t > 4): a lane's sparse tokens score / select / attend over the lane's C chunk cores, the
// select coordinated by the lane's own coordinator core (kernels/mla/dsa_coord.cpp)
#if defined(MLA_ATT_LANES)
constexpr uint32_t LANES = MLA_ATT_LANES;
#else
constexpr uint32_t LANES = 1;
#endif
// tree role of chunk core c at level L (C a power of two)
inline bool recv_at(uint32_t c, uint32_t L, uint32_t C) { return (c & ((2u << L) - 1)) == 0 && c + (1u << L) < C; }
inline bool send_at(uint32_t c, uint32_t L) { return (c & ((2u << L) - 1)) == (1u << L); }
// row tiles of token t taken by chunk core c: jt = c + i C, i < nk
inline uint32_t chunk_tiles(uint32_t p, uint32_t c, uint32_t C) {
    const uint32_t n = p / 32 + 1;
    return c < n ? (n - c + C - 1) / C : 0;
}
}  // namespace mlaatt

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace mlaatt {
// THE VISIBLE-PREFIX READ (decode lever, coordinator 2026-10-24 04:01:08Z): a page-table row is read only over the
// entries its token can touch (n_entries, from the position), 64-B granular, at most the row's page. A full-row read
// moved a CAPACITY-sized row per token and pass on every owner core (the pair's 1M local table: 32 KB, three reads per
// token on the sharded line = +42 us per DSA layer-step, doc/ttnn/14-production-tp8-megakernel/w1gap). The entries read
// are the same bytes and no consumer indexes past them: bitwise by construction.
template <typename Acc>
FORCE_INLINE void read_pt_prefix(const Acc& pt, uint32_t row, uint32_t dst, uint32_t n_entries) {
    const uint32_t page = pt.get_aligned_page_size(), want = (4 * n_entries + 63) & ~63u;
    noc_async_read(pt.get_noc_addr(row), dst, want < page ? want : page);
}
}  // namespace mlaatt
#endif
