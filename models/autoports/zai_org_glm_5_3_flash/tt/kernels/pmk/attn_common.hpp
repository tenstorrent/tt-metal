// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill megakernel (PMK) — the sparse_dsa ATTENTION phase of one layer program (tt/pmk_attn.py; the design page is
// doc/ttnn/16-prefill-production/prefill_megakernel.md, the contract its §4b / §4c).
//
// Derived from Stage 14's decode attention kernels (kernels/mla/att_{dm0,c,dm1}.cpp, att_dsa_*.cpp, revs 29-55): the
// same online-softmax partial (fp32 stats m / l, O rescaled by diagonal coefficient tiles on the FPU), the same CB
// discipline (every batch pads its CBs to NK so the compute's page-index reads start at the CB base). For prefill:
//   * the unit of work is a WORK ITEM (head h, q tiles qt0 .. qt0 + QG - 1): QG x 32 query rows of ONE head, so every
//     K tile read serves QG query tiles (rev 3: the silicon unit of rev 2 was DRAM-bound at 1 query tile per K read,
//     64 FLOP/B -- README §7). q_lat [1, H_r, C, 512] bf16 TILE, page (h * CT + qt) * DT + d; Q is read ONCE per item;
//   * an item attends ALL its keys on one core (no partial tree), keys in batches of NK tiles; per q tile j of the item
//     one running partial (stats m / l + O) updated IN PLACE (the TRISC seeks its CB pointers to region j before each
//     batch: reads and writes alias, group by group -- the reads of a DST group precede its pack);
//   * keys of an item = [context tiles] ++ [chunk tiles 0 .. qt0 + QG - 1]; a batch never mixes context and chunk
//     tiles; chunk tile k applies to q tile qt iff k <= qt (k == qt: the causal triangle, CB_AID page 0, no mask
//     traffic); context tiles apply to every q tile (dense) or carry per-(tile, q tile) membership masks (sparse).
//   * Sources: context from the slot's paged latent cache (dense) or gathered from the ROW_MAJOR latent store by
//     selected pool (sparse, 8 pools x 4 rows per K tile, tilized by the TRISC); the chunk's own rows from the
//     gathered chunk latent [1, 1, C, 512] -- an L1-SHARDED store over the MM grid in the layer program (the AG's
//     landing store, §4b H2), so chunk K reads are NoC L1 -> L1, not DRAM (the accessor resolves either).
// Sparse <=> chunk_start + C > 2051 (every query of the chunk selects; rows with <= 512 closed pools select every
// visible pool: exact, doc/architecture.md §4).
// Rev 4 (README §8): the chunk batches of an item are read in a per-core ROTATED order (batch_order: rotation
// (core + item) mod nb) so the 104 readers spread over the store's owner cores instead of hitting the same NK owners
// in lockstep (rev 3's 79 % compute wait); the sparse pool gather reads the row-major store one page (row) at a time;
// the mask constant is NEG_MASK = -1e9.
// Rev 7 (README §9): sparse masks as pool words + bf16 rows (the BRISC 56-59 % / NCRISC 64-67 % of a 11,971-us unit:
//   a Tensix RISC pays ~8-10 cycles per L1 load / store, so 1 RMW per id and 512 stores per mask tile both bound);
//   dense (PMK_ATT_DUAL_NOC): the chunk-tile reads of a batch are split over both NoCs -- the BRISC reads the even
//   tiles on NoC0, the NCRISC the odd ones on NoC1 (CB_AGO pages carry the slot; the NCRISC raises the slot's flag
//   word in CB_ASCR after its barrier): 4,773 us (QG 2: 4,100) under the 6,383 bar.
// Rev 8 (README §10): no per-id and no per-element RISC work on the sparse path.
//   * the selection arrives as per-row BITMAPS (sel [1, 1, C, W] uint32 ROW_MAJOR: bit q = pool q selected, context
//     pools then the chunk's 512; W = sel_row_words); the NCRISC burst-reads the item's 32 QG rows into its CB_ABM
//     page and ORs them into the union bitmap (word ops only);
//   * K tiles are FIXED 8-pool groups (group g = pools 8 g .. 8 g + 7; gathered iff the union touches it), so a row's
//     membership byte for a tile is ONE byte of its bitmap (mask_byte / mask_byte_diag);
//   * the BRISC writes a mask operand as 32 words: column 0 of a bf16 tile = the row's membership byte as an integer
//     (mask_col_row; the ring pages are zeroed once per phase);
//   * the TRISC expands it: one tile matmul B_col @ E8 (E8 row 0 = 2^-s on pool s's 4 columns, CB_AID page 2) gives
//     byte x 2^-s per lane exactly, then one SFPI pass (attn_mask_sfpu.hpp) adds the exact fp32 -1e9 where
//     floor(.) is even, the scale multiply fused in. The diagonal tile keeps the causal triangle add.
#pragma once

#include <cstdint>

#if !defined(FORCE_INLINE)
#define PMKATT_HOST 1  // host builds of the pure helpers (tests/s16/pmk/attn_mask_host.cpp)
#define FORCE_INLINE inline
#endif

namespace pmkatt {
constexpr uint32_t DT = 16;         // latent tiles (512 / 32)
constexpr uint32_t STAT_TILES = 2;  // m, l
constexpr uint32_t KPOOL = 4;       // tokens per DSA pool
constexpr uint32_t POOLS_PER_TILE = 8;
constexpr uint32_t TOPK = 512;         // selected pools per query row
constexpr uint32_t CHUNK_POOLS = 512;  // pools of a 2,048-row chunk (C / KPOOL; CT <= 64)
constexpr uint32_t DENSE_MAX = 2051;
constexpr uint32_t NONE_ID = 0xFFFFFFFFu;        // ids padding (fewer than 512 visible pools)
constexpr uint32_t NEG_MASK_BITS = 0xCE6E6B28u;  // -1e9f (tt/pmk_attn.py NEG_MASK; the triangle tile, dense diagonal)
constexpr uint32_t NEG16 = 0xCE6Eu;  // bf16(-1e9) = -998,244,352 (the bf16 triangle tile; pmk_attn.NEG_MASK_BF16)
// K tiles per batch (CB_AK = KSLOTS x NK x DT pages: KSLOTS - 1 batches in flight while one computes). tt/pmk_attn.py
// passes PMK_ATT_NK always.
#if defined(PMK_ATT_NK)
constexpr uint32_t NK = PMK_ATT_NK;
#else
constexpr uint32_t NK = 2;
#endif
// q tiles per item (tt/pmk_attn.py geo.qg; PMK_QG always passed)
#if defined(PMK_QG)
constexpr uint32_t QG = PMK_QG;
#else
constexpr uint32_t QG = 1;
#endif
// CB_AK slots (rev 7 knob geo.kslots -> PMK_ATT_KSLOTS; 2 = the rev 3..6 double buffer)
#if defined(PMK_ATT_KSLOTS)
constexpr uint32_t KSLOTS = PMK_ATT_KSLOTS;
#else
constexpr uint32_t KSLOTS = 2;
#endif
// plan page (CB_AINFO, 3 words) of a batch:
//   w0 = nk | kind << 8 | from_rm << 12  (kind: 0 = context tiles, 1 = chunk tiles; from_rm: K tiles in CB_AKRM)
//   w1 = k0 (chunk kind: the chunk tile index of the batch's first tile) | qg << 16 (q tiles of the item) | qt0 << 24
//        (the item's first q tile; CT <= 256)
//   w2 = last | mask_bits << 1          (last: the item's last batch; mask bit (i * QG + j): tile i has a mask tile
//                                        for q tile j in CB_AMASK, in (i, j) order; the dense path pushes none)
constexpr uint32_t KIND_CTX = 0, KIND_CHUNK = 1;
constexpr uint32_t KIND_END = 0xE1Du;  // w0 of the item-list terminator (no more items on this core)
inline uint32_t plan_w0(uint32_t nk, uint32_t kind, bool from_rm) {
    return nk | (kind << 8) | ((from_rm ? 1u : 0u) << 12);
}
// CB_AGO page (rev 7 dual-NoC dense reads, BRISC -> NCRISC, 4 words): w0 = the CB_AK slot's L1 address (GO_END: no
// more batches), w1 = k0, w2 = nk, w3 = the slot's flag index (CB_ASCR word FLAG_OFF / 4 + w3; the NCRISC stores 1
// there after its reads landed, the BRISC cleared it before pushing the page)
constexpr uint32_t GO_END = 0xFFFFFFFFu;
// rev 10: w4 = the source of the tiles (0 = the chunk latent store, tile k waits the LATB counter; 1 = the context
// L1 store, tile k = context tile k, landed before the phase); an ITEM_END page (w0) closes every item's pages, so
// the NCRISC needs no batch count (the sparse union is data-dependent)
constexpr uint32_t ITEM_END = 0xFFFFFFFEu;
constexpr uint32_t SRC_CHUNK = 0, SRC_CTX = 1;

// the rotation of an item's nb chunk batches on this core (tt/pmk_attn.py batch_order: (core + it) mod nb; nb <= 64)
inline uint32_t batch_rot(uint32_t core, uint32_t it, uint32_t nb) {
    uint32_t r = core + it;
    while (r >= nb) {
        r -= nb;
    }
    return r;
}

// byte offset of row r's left 16-element face row inside a 32 x 32 bf16 tile (the right face is + 512)
inline uint32_t row_off(uint32_t r) { return (r >> 4) * 1024 + (r & 15) * 32; }
// row r, column c of a 32 x 32 fp32 tile: word index
inline uint32_t f32_idx(uint32_t r, uint32_t c) { return (r >> 4) * 512 + (c >> 4) * 256 + (r & 15) * 16 + (c & 15); }

// ---------------------------------------------------------------- rev 8 selection bitmaps (the CB_ABM page of an item)
// sel row = (ctx_pools + CHUNK_POOLS) bits, 64-B rounded: words per row; the item's page = [32 QG rows x row words]
// [union bitmap over the context pools: ctxw words]
inline uint32_t sel_row_words(uint32_t ctx_pools) { return ((ctx_pools + CHUNK_POOLS + 511) / 512) * 16; }
inline uint32_t abm_uni_words(uint32_t ctx_pools) { return (ctx_pools + 31) / 32; }
inline uint32_t abm_words(uint32_t ctx_pools, uint32_t qg) {
    return 32 * qg * sel_row_words(ctx_pools) + abm_uni_words(ctx_pools);
}

// ---------------------------------------------------------------- rev 8 mask operand
// Row r's membership byte for the diagonal chunk tile (tile k == q tile qt, row r at position p = base + r, base a
// multiple of 4): with so = r >> 2, c = r & 3: pools s < so keep their membership; pool so is the pool closed at p
// when c == 3 (membership) and the OPEN pool otherwise (allowed: the tail [p + 1 - (p + 1) % 4, p] == [4 so, p]; the
// TRISC's causal triangle masks its tokens > p); pools s > so: allowed here, masked by the triangle (tokens > p).
inline uint32_t mask_byte_diag(uint32_t b, uint32_t r) {
    const uint32_t so = r >> 2, c = r & 3;
    const uint32_t keep = (c == 3 ? (2u << so) : (1u << so)) - 1u;  // membership bits kept
    return (b & keep) | (0xFFu & ~keep);
}
// bf16 bits of the integer n in [0, 255] (exact: 8 significant bits)
inline uint32_t bf16_of_byte(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    const uint32_t e = 31u - static_cast<uint32_t>(__builtin_clz(n));  // floor(log2 n), 0..7
    return ((127u + e) << 7) | ((n << (7u - e)) & 0x7Fu);
}
// word index of (row r, columns 0-1) in a 32 x 32 bf16 tile (columns 0 .. 15 of rows 0 .. 15 are face 0, rows
// 16 .. 31 face 2; a face row is 8 words)
inline uint32_t col0_word(uint32_t r) { return (r >> 4) * 256 + (r & 15) * 8; }
// the mask operand's row r: column 0 = the byte as a bf16 integer, column 1 = 0 (the other 30 columns stay zero)
template <class W>
FORCE_INLINE void mask_col_row(W m, uint32_t r, uint32_t byte) {
    m[col0_word(r)] = bf16_of_byte(byte);
}
}  // namespace pmkatt

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
namespace pmkatt {
// TRISC: both CB pointers to page `page` of the CB (the partial regions: region j of q tile j; reads and writes then
// alias -- the in-place online-softmax update). The CB must be quiescent for this RISC (its pushes / pops balanced
// per batch). Addresses in LocalCBInterface are 16-B words.
FORCE_INLINE void cb_seek(uint32_t cb, uint32_t page) {
    UNPACK(({
        LocalCBInterface& i = get_local_cb_interface(cb);
        i.fifo_rd_ptr = i.fifo_limit - i.fifo_size + page * i.fifo_page_size;
    }));
    PACK(({
        LocalCBInterface& i = get_local_cb_interface(cb);
        i.fifo_wr_ptr = i.fifo_limit - i.fifo_size + page * i.fifo_page_size;
        i.fifo_wr_tile_ptr = 0;
    }));
}
}  // namespace pmkatt
#endif

#if !defined(PMKATT_HOST) && !defined(COMPILE_FOR_TRISC)
// ---------------------------------------------------------------- rev 8 item bitmaps (a data-movement RISC, the
// NCRISC) The item's 32 qg selection rows (sel pages qt0 * 32 ..) into the CB_ABM page, one read burst, then the union
// bitmap over the context pools (OR of the rows' first ctxw words): word operations only.
#include "api/dataflow/dataflow_api.h"
namespace pmkatt {
template <class Acc>
FORCE_INLINE void fetch_item_bitmaps(uint32_t page, const Acc& sel, uint32_t qt0, uint32_t qg, uint32_t ctx_pools) {
    const uint32_t rw = sel_row_words(ctx_pools), nuni = abm_uni_words(ctx_pools), nrows = 32 * qg;
    for (uint32_t r = 0; r < nrows; ++r) {
        noc_async_read_page(qt0 * 32 + r, sel, page + r * rw * 4);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page);
    volatile tt_l1_ptr uint32_t* uni = rows + nrows * rw;
    for (uint32_t w = 0; w < nuni; ++w) {
        uint32_t acc = 0;
        for (uint32_t r = 0; r < nrows; ++r) {
            acc |= rows[r * rw + w];
        }
        uni[w] = acc;
    }
    invalidate_l1_cache();
}
}  // namespace pmkatt
#endif
