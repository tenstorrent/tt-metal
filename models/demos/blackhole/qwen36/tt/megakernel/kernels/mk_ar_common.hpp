// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 megakernel AR track (DESIGN_megakernel.md §6): CB / semaphore ids and the x-delivery handshake.
//
// One AR = one TP all-reduce of the block output (CB_OUT, 2 tiles per STREAM core = this core's 64 columns of the
// [32, 5120] row-parallel partial) into the next block's input (CB_X, the full [32, 5120] replicated on every STREAM
// core), with the residual add fused:
//   1. every STREAM core NOC-writes its CB_OUT shard into the sender core's staging buffer (slice idx) and bumps the
//      sender's producer counter (SEM_AR_PROD);
//   2. the sender (mk_ar_sender.cpp, 2 links) multicasts slice c over the fabric into STREAM core c's receive slot
//      `ring index` on every device, plus the local copy; each slot write bumps that core's receive semaphore;
//   3. each STREAM core sums its 8 slots in ring order and adds its residual slice (CB_AR_RES -> CB_AR_SEND, FP32 DST),
//      giving bit-identical replicas of the new residual on every device;
//   4. it unicasts the new 64-column slice into its x root's CB_X at the slice's final offset and bumps that root's
//      counter; root k (STREAM cores 76..79, slices [20k, 20k + 20)) multicasts its 80 KB range into CB_X on all 80
//      STREAM cores (roots 0/1 on NOC0 from BRISC, 2/3 on NOC1 from NCRISC, as fm11's h broadcast) and publishes flag
//      k = seq + 1; a core's CB_X is complete when all 4 flags reach seq + 1. (80 concurrent multicasts into
//      overlapping rectangles stalled for ~150 us in fused-MLP v1/v2; 4 roots took ~5 us for 136 KB.)
// The RMSNorm is left to the consumer (norm-ready x: the full row is on every core, so the consumer can take the row
// statistic itself and fold gamma into its weights); the new residual slice stays in CB_AR_SEND for the next AR.
//
// Semaphores: tt-metal has only 16 program semaphores per core (fm11 uses 0-11), so the AR track uses GLOBAL
// semaphores (L1 words at one address on every core and device; addresses in runtime args): the receive semaphore
// per site, 4 x-root counters and 4 x-root flags. The sender core's producer counter is program semaphore SEM_AR_PROD
// on the sender core only.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// (MKAR_ALL_RT = the engine includes this file: its own CB_X / CB_OUT / ... names stay in charge)
#if !defined(MKAR_ALL_RT)
constexpr uint32_t CB_X = 0, CB_OUT = 16;
constexpr uint32_t CB_AR_SEND = 17;  // new residual slice (2 tiles): AR output, next AR's residual input
constexpr uint32_t CB_AR_RECV = 18;  // receive slots: ring x 2 tiles
constexpr uint32_t CB_AR_RES = 19;   // residual slice in (2 tiles)
constexpr uint32_t SEM_AR_PROD = 0;  // program semaphore on the (non-STREAM) sender core: producer slices staged
#endif
constexpr uint32_t AR_NROOT = 4, AR_SLICES_PER_ROOT = 20;

struct ArRect { uint32_t x0, y0, x1, y1, n; };

FORCE_INLINE uint64_t ar_mcast_addr(const ArRect& r, uint32_t l1, uint8_t noc) {
    return noc == 0 ? get_noc_multicast_addr(r.x0, r.y0, r.x1, r.y1, l1, noc)
                    : get_noc_multicast_addr(r.x1, r.y1, r.x0, r.y0, l1, noc);
}
FORCE_INLINE void ar_bcast(uint32_t src, uint32_t dst, uint32_t bytes, const ArRect* r, uint32_t self_rect, uint8_t noc) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (r[i].n == 0) continue;
        const uint64_t a = ar_mcast_addr(r[i], dst, noc);
        if (i == self_rect) {
            // the source slice sits at a different L1 address than its CB_X slot, so the loopback copy is a real copy
            noc_async_write_multicast_loopback_src(src, a, bytes, r[i].n, false, noc);
        } else {
            noc_async_write_multicast(src, a, bytes, r[i].n, false, noc);
        }
    }
}
FORCE_INLINE void ar_flag(uint32_t flag_l1, const ArRect* r, uint32_t self_rect, uint8_t noc) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (r[i].n == 0) continue;
        const uint64_t a = ar_mcast_addr(r[i], flag_l1, noc);
        if (i == self_rect) noc_semaphore_set_multicast_loopback_src(flag_l1, a, r[i].n, false, noc);
        else noc_semaphore_set_multicast(flag_l1, a, r[i].n, false, noc);
    }
}
FORCE_INLINE volatile tt_l1_ptr uint32_t* ar_gsem(uint32_t addr) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }
#if !defined(MKAR_ALL_RT)
// root k: wait for its 20 slices of call `seq`, multicast its CB_X range everywhere, publish flag k = seq + 1
FORCE_INLINE void ar_root_bcast(uint32_t k, uint32_t seq, uint32_t chunk, uint32_t cnt_addr, uint32_t flg_addr,
                                const ArRect* r, uint32_t self_rect, uint8_t noc) {
    noc_semaphore_wait_min(ar_gsem(cnt_addr), AR_SLICES_PER_ROOT * (seq + 1));
    const uint32_t off = get_write_ptr(CB_X) + k * AR_SLICES_PER_ROOT * chunk;
    ar_bcast(off, off, AR_SLICES_PER_ROOT * chunk, r, self_rect, noc);
    noc_async_write_barrier(noc);
    *ar_gsem(flg_addr) = seq + 1;
    ar_flag(flg_addr, r, self_rect, noc);
    noc_async_write_barrier(noc);
}
#endif
