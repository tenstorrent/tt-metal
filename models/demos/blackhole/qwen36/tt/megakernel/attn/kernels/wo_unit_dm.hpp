// Includable wo unit, dataflow side (spec v1.5 / v1.8a). One wo core computes nt (<= WO_MAXT) output tiles of
// out[M, N] = A[M, WK] (.) S? @ W[WK, N] (split-N); the owners collect the output by chunks of 10 tiles.
// Used by the MKA2 EXT role (attention wo; GDN24 bring-up on the 16 owners) and meant for the GDN role binary
// (gdn_out: WK = 24, S off, A written by the 24 GDN cores, ready word GS_GDNA).
//
// Weight slice of one wo core: k-major pages (kl * nt + j, kl < WK), contiguous in one DRAM bank. The page size is the
// ring CB's tile size (BFP8 wo 1088 B, BFP4 gdn_out 576 B), so the same code streams either format.
// Output tiles: bf16 (the CB_QO / owner CB_XO format); owner chunk c = output tiles 10c .. 10c + 9.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace wou {

constexpr uint32_t CHUNK = 10;   // output tiles per owner chunk (owner j holds tiles 10j .. 10j + 9)

// Stream pages [p0, p1) of a weight slice into the ring CB at its write pointer (caller reserved the full slot):
// `chunk`-page reads, at most `ninflight` chunks in flight on transaction id `trid`; gate() runs before each batch
// (e.g. the v12 busy gate). Page bytes come from the ring CB's format.
template <typename Gate>
FORCE_INLINE void ring_fill(uint32_t cb_ring, uint64_t src, uint32_t p0, uint32_t p1, uint32_t chunk, uint32_t ninflight,
                            uint32_t trid, Gate gate) {
    const uint32_t pb = get_tile_size(cb_ring), dst = get_write_ptr(cb_ring);
    noc_async_read_set_trid(trid);
    for (uint32_t p = p0, c = 0; p < p1; p += chunk, ++c) {
        if (c && (c % ninflight) == 0) { noc_async_read_barrier_with_trid(trid); noc_async_read_set_trid(trid); }
        if ((c % ninflight) == 0) gate();
        const uint32_t n = (p1 - p) < chunk ? (p1 - p) : chunk;
        noc_async_read(src + p * pb, dst + p * pb, n * pb);
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier_with_trid(trid);
}

// Scatter this core's nt output tiles (front of cb_out, `slot` pages reserved per phase) to the owners' chunk buffers
// (uniform L1 address xo on every owner), then += count on each owner's counter word. tile[j] = global output tile.
// kVc (GOTAIL E1_GO_SCVC): the writes + count incs on that unicast VC (default: the API default, i.e. unchanged code).
template <uint32_t kVc = NOC_UNICAST_WRITE_VC>
FORCE_INLINE void scatter(uint32_t cb_out, uint32_t slot, uint32_t nt, const uint32_t* tile, const uint32_t* ow_x,
                          const uint32_t* ow_y, uint32_t xo, uint32_t cnt_word) {
    const uint32_t tb = get_tile_size(cb_out), src = get_read_ptr(cb_out);
    uint32_t cnt[16];
    for (uint32_t i = 0; i < 16; ++i) cnt[i] = 0;
    for (uint32_t j = 0; j < nt; ++j) {
        const uint32_t ch = tile[j] / CHUNK;
        noc_async_write(src + j * tb, get_noc_addr(ow_x[ch], ow_y[ch], xo + (tile[j] % CHUNK) * tb), tb, noc_index, kVc);
        ++cnt[ch];
    }
    noc_async_write_barrier();
    for (uint32_t ch = 0; ch < 16; ++ch)
        if (cnt[ch]) noc_semaphore_inc(get_noc_addr(ow_x[ch], ow_y[ch], cnt_word), cnt[ch], noc_index, kVc);
    cb_pop_front(cb_out, slot);
}

// GOTAIL E1_GO_SCPASS: as scatter(), but the output tiles arrive in DST passes of nw (matmul_nt_ks<., ., true>): each
// pass's tiles are written as soon as they are pushed (the next pass computes meanwhile); one barrier + the incs at the end.
template <uint32_t kVc = NOC_UNICAST_WRITE_VC>
FORCE_INLINE void scatter_ps(uint32_t cb_out, uint32_t slot, uint32_t nt, const uint32_t* tile, const uint32_t* ow_x,
                             const uint32_t* ow_y, uint32_t xo, uint32_t cnt_word, uint32_t nw) {
    const uint32_t tb = get_tile_size(cb_out), src = get_read_ptr(cb_out);
    uint32_t cnt[16];
    for (uint32_t i = 0; i < 16; ++i) cnt[i] = 0;
    for (uint32_t j0 = 0; j0 < nt; j0 += nw) {
        const uint32_t j1 = (nt - j0) < nw ? nt : j0 + nw;
        cb_wait_front(cb_out, j1);
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t ch = tile[j] / CHUNK;
            noc_async_write(src + j * tb, get_noc_addr(ow_x[ch], ow_y[ch], xo + (tile[j] % CHUNK) * tb), tb, noc_index, kVc);
            ++cnt[ch];
        }
    }
    cb_wait_front(cb_out, slot);   // the padding push (nt < slot) before the pop
    noc_async_write_barrier();
    for (uint32_t ch = 0; ch < 16; ++ch)
        if (cnt[ch]) noc_semaphore_inc(get_noc_addr(ow_x[ch], ow_y[ch], cnt_word), cnt[ch], noc_index, kVc);
    cb_pop_front(cb_out, slot);
}

}  // namespace wou
