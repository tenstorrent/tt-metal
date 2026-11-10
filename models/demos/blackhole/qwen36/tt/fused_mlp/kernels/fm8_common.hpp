// Fused MLP v8: CB ids, semaphores and the multi-root broadcast helpers shared by the RISCs.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_W2 = 5, CB_OUT = 16;
constexpr uint32_t NROOT = 4;
// semaphores: x counters 0-3, x flags 4-7, h counters 8-11, h flags 12-15
constexpr uint32_t SEM_XCNT0 = 0, SEM_XFLG0 = 4, SEM_HCNT0 = 8, SEM_HFLG0 = 12;

struct Rect { uint32_t x0, y0, x1, y1, n; };

// multicast [src, src+bytes) to the same L1 address on both rectangles; noc 1 takes the reversed corner order
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
// root side: wait for `expect` producer increments, multicast the chunk, then raise this root's flag everywhere
FORCE_INLINE void root_bcast(uint32_t cnt_sem, uint32_t flg_sem, uint32_t expect, uint32_t chunk_l1, uint32_t bytes,
                             const Rect* r, uint32_t self_rect, uint8_t noc) {
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(cnt_sem));
    volatile tt_l1_ptr uint32_t* flg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flg_sem));
    noc_semaphore_wait(cnt, expect);
    *cnt = 0;
    bcast_rects(chunk_l1, bytes, r, self_rect, noc);
    noc_async_write_barrier(noc);
    *flg = 1;
    flag_rects((uint32_t)flg, r, self_rect, noc);
    noc_async_write_barrier(noc);
}
FORCE_INLINE void wait_flags(uint32_t flg0) {
    for (uint32_t i = 0; i < NROOT; ++i)
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flg0 + i)), 1);
}
FORCE_INLINE void clear_flags(uint32_t flg0) {
    for (uint32_t i = 0; i < NROOT; ++i) *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flg0 + i)) = 0;
}
