// Fused MLP v11 (persistent, multi-layer): CB ids, semaphores and multi-root broadcast helpers.
// Layer L's h flag value is L + 1 (cumulative), h counters are cumulative, and every core reports "down done"
// to the 4 roots so a root never overwrites the h landing buffers while a core still reads layer L - 1's h.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_W2 = 5, CB_OUT = 16;
constexpr uint32_t NROOT = 4;
// semaphores: h counters 0-3, h flags 4-7, down-done counters 8-11
constexpr uint32_t SEM_HCNT0 = 0, SEM_HFLG0 = 4, SEM_DONE0 = 8;

struct Rect { uint32_t x0, y0, x1, y1, n; };

FORCE_INLINE void bcast_rects(uint32_t l1, uint32_t bytes, const Rect* r, uint32_t self_rect, uint8_t noc) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (r[i].n == 0) continue;
        uint64_t a = noc == 0 ? get_noc_multicast_addr(r[i].x0, r[i].y0, r[i].x1, r[i].y1, l1, noc)
                              : get_noc_multicast_addr(r[i].x1, r[i].y1, r[i].x0, r[i].y0, l1, noc);
        if (i == self_rect) noc_async_write_multicast_loopback_src(l1, a, bytes, r[i].n, false, noc);
        else noc_async_write_multicast(l1, a, bytes, r[i].n, false, noc);
    }
}
FORCE_INLINE void flag_rects(uint32_t flag_l1, const Rect* r, uint32_t self_rect, uint8_t noc) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (r[i].n == 0) continue;
        uint64_t a = noc == 0 ? get_noc_multicast_addr(r[i].x0, r[i].y0, r[i].x1, r[i].y1, flag_l1, noc)
                              : get_noc_multicast_addr(r[i].x1, r[i].y1, r[i].x0, r[i].y0, flag_l1, noc);
        if (i == self_rect) noc_semaphore_set_multicast_loopback_src(flag_l1, a, r[i].n, false, noc);
        else noc_semaphore_set_multicast(flag_l1, a, r[i].n, false, noc);
    }
}
// root k, layer L: wait until every core finished layer L-1's down (done >= ncores * L), wait for this chunk's
// producers (cumulative count), multicast the chunk, then publish flag = L + 1 everywhere.
FORCE_INLINE void root_bcast_layer(uint32_t k, uint32_t L, uint32_t ncores, uint32_t expect_per_layer, uint32_t chunk_l1,
                                   uint32_t bytes, const Rect* r, uint32_t self_rect, uint8_t noc) {
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_DONE0 + k));
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HCNT0 + k));
    volatile tt_l1_ptr uint32_t* flg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLG0 + k));
    {
        DeviceZoneScopedN("ROOT-WAIT");
        noc_semaphore_wait_min(done, ncores * L);
        noc_semaphore_wait_min(cnt, expect_per_layer * (L + 1));
    }
    DeviceZoneScopedN("ROOT-MC");
    bcast_rects(chunk_l1, bytes, r, self_rect, noc);
#if defined(MK_FLAG_FLUSH) && MK_FLAG_FLUSH
    // -mk-tail: the flag multicast follows the data multicast on the same NOC from the same core; wait only until the
    // data has left this core (flushed), not for every receiver's ack, before publishing the flag.
    noc_async_writes_flushed(noc);
#else
    noc_async_write_barrier(noc);
#endif
    *flg = L + 1;
    flag_rects((uint32_t)flg, r, self_rect, noc);
    noc_async_write_barrier(noc);
}
FORCE_INLINE void wait_hflags(uint32_t L) {
    for (uint32_t i = 0; i < NROOT; ++i)
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_HFLG0 + i)), L + 1);
}
FORCE_INLINE void reset_sems() {
    for (uint32_t i = 0; i < 12; ++i) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i)) = 0;
}
