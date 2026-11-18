// MK_V12: the fm12 / v12 split-K MLP phase inside the MKA2 engine (spec v0.6-v0.9, -mk-v12 V12_MLP_RESULTS.md).
// STREAM core (n, k) of the 10 x 8 rect: gate|up W[K-slice k, h-slice n (nn = 7 / 6 pairs)] and down Wd[h-slice n,
// out-slice k], one contiguous 420-tile BFP4 slice per layer (CB_WV, issued by plan_layer with the layer's trid).
// Owners = STREAM_EXT cores (10 + slot, k): owner j = 2k + slot holds x chunk j; it multicasts x to K-row k (AR
// stand-in), sums the 10 column down partials (P @ I, fp32) into CB_XO and releases it.
// Column reduce-scatter: pair j is summed at row j (P @ I) and scaled by rstd (CB 20 stand-in, shared with RQKV),
// then SwiGLU; h is all-gathered by unicast (rows below on NOC0 from the writer, rows above on NOC1 from the reader).
// All words are global-semaphore L1 addresses (uniform), monotonic per layer (>= waits), reset at program end.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_WV = 32, CB_XK = 33, CB_VA = 34, CB_VOWN = 35, CB_VRS = 36, CB_VH = 37, CB_VHO = 38, CB_VHO2 = 39;
// ids 21-23 / 30 and the fm11 MLP's 3-5 (dropped under V12) / 15 keep the highest CB id at 41: the CB config block
// (16 B per id up to the highest) counts against the 69 KB kernel config buffer
constexpr uint32_t CB_VDO = 40, CB_VDO2 = 41, CB_VI = 42, CB_VRSA = 43, CB_VRT = 44, CB_VJ = 30;
constexpr uint32_t CB_VP = 3, CB_VXO = 4, CB_VX0 = 5, CB_VPK = 16;   // 15/21-23 are MKA_RWO's (A, S, AS, XO)
constexpr uint32_t V_NNM = 7, V_NTM = 14, V_KTR = 20, V_OT = 10, V_NROW = 8, V_NCOL = 10, V_PTB = 2048;
constexpr uint32_t V_SLOT = 3 * V_KTR * V_NNM;   // 420 tiles
constexpr uint32_t V_TRID0 = 12;   // MLP slice read trids 12 / 13 (layer parity)
// v12 global words (index into the v12 address array)
#if defined(MK_V12_NOCBAL)
// MK80 NoC-balanced MLP slice (DRAM_CEILING.md / DESIGN_mk70 §9.1): the STREAM writer (NOC0) reads the first
// MK_V12_NOCBAL % of the slice, the reader (NOC1) the rest, each with 1 request of MK_V12_NOCBAL_PB bytes in flight.
// Writer -> reader mailbox per slice: VSL / VSH = NOC1 address of the reader's part, then VS = slice seq; reader -> writer:
// VD = seq once its part landed (the writer pushes CB_WV after it). Data movement only: bytes and L1 placement unchanged.
constexpr uint32_t VG_XA = 0, VG_XB = 1, VG_RS = 2, VG_H = 3, VG_P = 4, VG_MDONE = 5, VG_MGO = 6, VG_LA = 7, VG_LB = 8,
                   VG_BUSY = 9, VG_VS = 10, VG_VSL = 11, VG_VSH = 12, VG_VD = 13, NVG = 14;
#ifndef MK_V12_NOCBAL_ATT
#define MK_V12_NOCBAL_ATT 0
#endif
#ifndef MK_V12_NOCBAL_PB
#define MK_V12_NOCBAL_PB 9216
#endif
constexpr uint32_t V_NB_PB = MK_V12_NOCBAL_PB;
// NOC0 bytes of a `total`-byte slice (a whole number of pieces)
FORCE_INLINE uint32_t v_nb_n0(uint32_t total) { return total * MK_V12_NOCBAL / 100 / V_NB_PB * V_NB_PB; }
#if !defined(MKA_ROLE_EXT)   // only STREAM cores stream a slice: the EXT binary keeps today's code (text budget)
#define V_NB_ON 1
#else
#define V_NB_ON 0
#endif
#else
#define V_NB_ON 0
constexpr uint32_t VG_XA = 0, VG_XB = 1, VG_RS = 2, VG_H = 3, VG_P = 4, VG_MDONE = 5, VG_MGO = 6, VG_LA = 7, VG_LB = 8,
                   VG_BUSY = 9, NVG = 10;
#endif
constexpr uint32_t V_FACE_B = 512, V_ROW_B = 32, V_XPK_SLOT_B = 10 * 2 * 16 * V_ROW_B;
// spec §18 qkv split-K (MKA_QSK) through this unit: column c's N-list = qkv tiles first(c) .. + qsk_nt(c) - 1 (112 over
// 10 columns: 12, 12, 11 x 8), slot M (CB_WM) = 20 K tiles x nt BFP8 pages, k-major; GDN-type layers read QSK_GK K tiles
// (in-proj stand-in, ~5.9 MB over the 80 cores). The qkv RS shares VG_RS / VG_LA / VG_LB with the MLP RS: LA / LB carry
// 2L + 1 (qkv) and 2L + 2 (MLP); an owner row's VG_RS count adds 7 per attention layer.
#if defined(MKA_QSK)
#if !defined(MK_RS_QKV)
#error "MKA_QSK needs MK_RS_QKV (the owners' mixer x delivery, flag 2L + 1)"
#endif
#if defined(MK_REAL_AR) && !defined(MK_AR_DLV)
#error "MKA_QSK + MK_REAL_AR needs the reader-side AR delivery (MK_AR_DLV): it raises the mixer flag 2L + 3 (X2)"
#endif
#ifndef QSK_GK
#define QSK_GK 6
#endif
constexpr uint32_t QSK_SLOT = 240, QSK_KPR = 20;
FORCE_INLINE uint32_t qsk_nt(uint32_t col) { return col < 2 ? 12u : 11u; }
#define V_LV(L) (2 * (L) + 2)
#if defined(MK_GDN_INPROJ)
// real GDN in-proj (spec v1.78 G-a): every layer's mixer is a split-K + column RS on STREAM -> one mixer RS phase per
// layer. gdn_in = 65 tiles over the 10 columns (7 x 5 + 6 x 5), 20 K tiles per K-row, BFP4 in its own slot CB 15
// (shares slot M's turn: the next layer's slice starts once compute popped this one). Owned tile -> the GDN cores'
// staging CB 41 (tile of column c, K-row r -> GDN core (c, 8 + r % 2), slot r / 2); the Q root raises in_ready once
// 65 (g + 1) tiles landed.
constexpr uint32_t CB_WMG = 15, GIN_NT = 65, GIN_SLOT = 20 * 7;
#define V_NRS(L) (2 * (L) + 2)
#define QMIX(L, A) ((L) + 1)
#else
#define V_NRS(L) ((L) + 1 + aidx(L) + (is_attn(L) ? 1u : 0u))
#define QMIX(L, A) ((A) + 1)
#endif
#else
#define V_LV(L) ((L) + 1)
#define V_NRS(L) ((L) + 1)
#endif

FORCE_INLINE volatile tt_l1_ptr uint32_t* vw(uint32_t addr) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }
// R = 32: whole tiles; R <= 16: rows 0..R-1 = two face writes (face 0 at +0, face 1 at +512) per tile
FORCE_INLINE void v_tile_write(uint32_t src, uint64_t dst, uint32_t ntiles, uint32_t R) {
    if (R >= 32) { noc_async_write(src, dst, ntiles * V_PTB); return; }
    for (uint32_t t = 0; t < ntiles; ++t) {
        noc_async_write(src + t * V_PTB, dst + t * V_PTB, R * V_ROW_B);
        noc_async_write(src + t * V_PTB + V_FACE_B, dst + t * V_PTB + V_FACE_B, R * V_ROW_B);
    }
}
FORCE_INLINE uint32_t v_pk_off(uint32_t t, uint32_t f, uint32_t R) { return (2 * t + f) * R * V_ROW_B; }
// Shared split-K column reduce-scatter (spec §18; MLP now, qkv / GDN in-proj next). A column's N-list is `nitem`
// items of `ipt` tiles (MLP: (gate j, up j) pairs, ipt 2; qkv: single tiles, ipt 1) and the partial is item-major
// (item i at tiles i*ipt ..). Row r owns items r, r+8, ...; each sender has V_RSO tile slots in an owner's landing
// buffer (MLP 1 item x 2, qkv <= 2 items x 1). Sender k writes the items of owner rows [r0, r1) to slot k (item i at
// ((i / 8) * ipt)), then bumps each such owner's rs word once (an owner waits 7 per RS phase).
constexpr uint32_t V_RSO = 2;
FORCE_INLINE void v_rs_send(uint32_t part, uint32_t r0, uint32_t r1, uint32_t nitem, uint32_t ipt, const uint32_t* cy,
                            uint32_t cx, uint32_t rsbuf, uint32_t k, uint32_t R, uint32_t rs_word) {
    for (uint32_t r = r0; r < r1; ++r)
        for (uint32_t i = r; i < nitem; i += V_NROW)
            v_tile_write(part + i * ipt * V_PTB, get_noc_addr(cx, cy[r], rsbuf + (k * V_RSO + (i / V_NROW) * ipt) * V_PTB), ipt, R);
    noc_async_write_barrier();
    for (uint32_t r = r0; r < r1; ++r) noc_semaphore_inc(get_noc_addr(cx, cy[r], rs_word), 1);
}
#if defined(MK_VRS_NOC0)   // E1_VRS_NOC0: the writer sends both halves; one out-of-line copy (writer text budget)
__attribute__((noinline)) void v_rs_send_ni(uint32_t part, uint32_t r0, uint32_t r1, uint32_t nitem, uint32_t ipt, const uint32_t* cy,
                                            uint32_t cx, uint32_t rsbuf, uint32_t k, uint32_t R, uint32_t rs_word) {
    v_rs_send(part, r0, r1, nitem, ipt, cy, cx, rsbuf, k, R, rs_word);
}
#endif
// h all-gather: h_k (one tile) to column rows [r0, r1), then H += 1 on each
FORCE_INLINE void v_hu_send(uint32_t src, uint32_t r0, uint32_t r1, const uint32_t* cy, uint32_t cx, uint32_t hdst,
                            uint32_t R, uint32_t h_word) {
    for (uint32_t r = r0; r < r1; ++r) v_tile_write(src, get_noc_addr(cx, cy[r], hdst), 1, R);
    noc_async_write_barrier();
    for (uint32_t r = r0; r < r1; ++r) noc_semaphore_inc(get_noc_addr(cx, cy[r], h_word), 1);
}
// multicast a word value to up to 2 rects (no loopback; rect n == 0 skipped); rect = {x0, y0, x1, y1, n} NOC0 coords
FORCE_INLINE void v_mcast_word(uint32_t addr, uint32_t val, const uint32_t (*rc)[5], uint8_t noc) {
    *vw(addr) = val;
    for (uint32_t i = 0; i < 2; ++i) {
        if (rc[i][4] == 0) continue;
        const uint64_t m = noc == 0 ? get_noc_multicast_addr(rc[i][0], rc[i][1], rc[i][2], rc[i][3], addr, noc)
                                    : get_noc_multicast_addr(rc[i][2], rc[i][3], rc[i][0], rc[i][1], addr, noc);
        noc_semaphore_set_multicast(addr, m, rc[i][4], false, noc);
    }
}
FORCE_INLINE uint64_t v_mc_addr(const uint32_t* r, uint32_t l1, uint8_t noc) {
    return noc == 0 ? get_noc_multicast_addr(r[0], r[1], r[2], r[3], l1, noc) : get_noc_multicast_addr(r[2], r[3], r[0], r[1], l1, noc);
}
// MLP / AR busy window (spec v0.8.9): odd = busy. Owners set/clear it around their MLP (and, later, AR) windows;
// owner 0 mirrors it to the rows 8-9 ext cores (x 0-9 only, v0.9.1) so their PQGDN reads hold while it is odd.
// Called from the owner WRITER (BRISC): it must use the writer's own NOC (noc_index = NOC0). Issuing on NOC1 from
// BRISC shares the NCRISC's NOC1 command buffers / ack counters and hung the engine once the reader delivered x on NOC1.
FORCE_INLINE void v_busy(uint32_t busy_word, uint32_t val, bool mirror, const uint32_t (*rc)[5]) {
    if (mirror) { v_mcast_word(busy_word, val, rc, noc_index); noc_async_write_barrier(); }
    else *vw(busy_word) = val;
}
#if V_NB_ON
// the STREAM reader's NOC1 part: wait for CB_VA (the compute's partial) while streaming its share of the current slice,
// 1 request in flight (TRID_NB), VD = seq once landed. One out-of-line copy for both call sites (text budget).
struct VNb { uint32_t seq, rem, dst, dst0, len; const uint32_t* vg; uint64_t src; };
constexpr uint32_t TRID_NB = 11;
__attribute__((noinline)) void v_nb_wait(VNb& b, uint32_t cb, uint32_t n) {
    while (!cb_pages_available_at_front(cb, n)) {
        if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, TRID_NB)) continue;
        if (!b.rem) {   // previous part landed (or none yet): publish, then look for the next slice (VS += 1 per slice)
            *vw(b.vg[VG_VD]) = b.seq;
            invalidate_l1_cache();
            if (*vw(b.vg[VG_VS]) == b.seq) continue;
            ++b.seq;
            b.src = ((uint64_t)*vw(b.vg[VG_VSH]) << 32) | *vw(b.vg[VG_VSL]);
            b.dst = b.dst0; b.rem = b.len;
        }
        const uint32_t sz = b.rem < V_NB_PB ? b.rem : V_NB_PB;
        noc_async_read_set_trid(TRID_NB);
        noc_async_read<V_NB_PB>(b.src, b.dst, sz);
        noc_async_read_set_trid(0);
        b.src += sz; b.dst += sz; b.rem -= sz;
    }
}
#endif
