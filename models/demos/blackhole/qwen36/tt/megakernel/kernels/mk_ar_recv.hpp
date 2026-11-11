// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR, AR_RECV (owner-core dataflow) as a callable phase (AR_INTERFACE.md): one all-reduce per iter(it), it =
// the AR index within the pass (monotonic semaphores compare against it + 1). Used by the standalone persistent kernel
// (mk_ps_receiver.cpp: init + ITERS x iter + fini) and by the engine's owner writer (call iter(L) at the AR_RECV hook).
//
// Compile-time config (macros, define before including):
//   MKAR_CB_RED, MKAR_CB_OUT, MKAR_CB_RES          reduction slots / reduced chunk (AG payload) / residual CBs
//   MKAR_FUSE_RES, MKAR_RES_TILES, MKAR_CHUNK_TILES, MKAR_IN_PAGE, MKAR_RING
//   MKAR_SEM_AG_READY, MKAR_SEM_DONE, MKAR_SEM_READY   (program-semaphore ids, or global-semaphore addresses with MK_PS_GSEM)
//   MKAR_PROG, MKAR_ROWS, MKAR_DELIVER, MKAR_STATS, MKAR_SRED, MKAR_AGD, MKAR_INPK
//   MKAR_CB_PART, MKAR_CB_SCALER, MKAR_CB_STAGE      Sum(x^2) CBs (STATS)
//   MKAR_ARGS_L1 (engine): L1 address of this core's arg table; carg / rarg index it instead of the runtime args
//     (runtime-arg space is sized by the largest core of the program, spec v1.25)
//   MKAR_DLV_SIGNAL (engine): L1 address of a local word; instead of delivering x itself, post it = it + 1 when the
//     output chunk is complete (the engine's owner reader multicasts it on NOC1)
//   MKAR_TRID (engine): transaction id for this phase's own-chunk reads (barrier on it, not on all reads)
//   MKAR_DLV_NONE (engine, MK_AR_GAMMA): no delivery and no post here (the engine scales the output first)
//   MKAR_ALL_RT: 1 = the "common" block is read from the runtime args at cbase (engine), 0 = from the common args
//   MKAR_SEM_BASE (with MK_PS_GSEM, v0.9.14): uniform L1 address of a persistent per-core word holding how many RS / AG
//     arrival increments earlier programs consumed. The arrival semaphore is then NEVER reset: trace replays are not
//     synchronised across the mesh, so a peer can already be in the next replay and increment this core's arrival
//     semaphore before this core's fini; a reset would wipe that increment and the next replay would wait forever.
//     Targets become base + per-program target; fini advances the base instead of zeroing the semaphore.
// Runtime layout: common block [sem, in_addr, n_senders, sender xy..., (AGD) agstage]
//                 role block   [role, wait_value, (owner) my_chip, n_link_senders, link sender xy..., n_segs, segs (3 each),
//                               (SRED) flag_base + order[ring], (DELIVER) xdst, 2 rects, xflag, (STATS) rslot, rtag]
#pragma once
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "mk_ar_common.hpp"
#include <cstdint>
#ifndef MK_PS_SEM
#ifdef MK_PS_GSEM
#define MK_PS_SEM(id) (id)
#else
#define MK_PS_SEM(id) get_semaphore(id)
#endif
#endif
#ifndef MKAR_ALL_RT
#define MKAR_ALL_RT 0
#endif
// MK_PS_PROGW (debug): progress word of this RISC at MK_PS_PROGW + 16 x 2 = (it << 8) | step
#ifndef PW
#ifdef MK_PS_PROGW
#define PW(slot, it, step) (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>((MK_PS_PROGW) + 16 * (slot)) = ((it) << 8) | (step))
#else
#define PW(slot, it, step)
#endif
#endif

// MKAR_LEANZ (engine, with MKA_LEAN_C): no AR-phase profiler zones (each costs code on the owner role, which is the
// text-sizing group once QSK + RWO land)
#ifndef MKAR_ZONE
#if defined(MKAR_LEANZ)
#define MKAR_ZONE(n)
#else
#define MKAR_ZONE(n) DeviceZoneScopedN(n)
#endif
#endif

namespace mkar {

// wrap-safe ">=" wait (v0.9.14): with never-reset arrival semaphores the counts wrap after 2^32 increments (~13 h of
// serving); compare the signed distance instead of the raw values
FORCE_INLINE void wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t target) {
#ifdef MKAR_SEM_BASE
    do {
        invalidate_l1_cache();
    } while (static_cast<int32_t>(*p - target) < 0);
#else
    noc_semaphore_wait_min(p, target);
#endif
}

constexpr uint32_t kRoleNone = 0, kRoleOwner = 2, XCT = 10;

struct Recv {
    uint32_t cbase = 0, rbase = 0;
    uint32_t role = 0, wait_value = 0, my_chip = 0, num_senders = 0, num_segs = 0;
    size_t sender_arg_base = 0, seg_base = 0, after_segs = 0, tail_base = 0;
    volatile tt_l1_ptr uint32_t* sem = nullptr;
    uint32_t in_addr = 0;
    uint32_t base = 0;        // arrival increments consumed by earlier programs (MKAR_SEM_BASE)
    uint32_t done_iters = 0;  // ARs of this program (fini advances the base by wait_value x done_iters)

    FORCE_INLINE uint32_t carg(uint32_t i) const {
#if defined(MKAR_ARGS_L1)   // engine (spec v1.25): the whole arg block lives in a per-core L1 table at MKAR_ARGS_L1
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_ARGS_L1)[cbase + i];
#endif
        if constexpr (MKAR_ALL_RT) {
            return get_arg_val<uint32_t>(cbase + i);
        } else {
            return get_common_arg_val<uint32_t>(i);
        }
    }
    FORCE_INLINE uint32_t rarg(size_t i) const {
#if defined(MKAR_ARGS_L1)
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_ARGS_L1)[i];
#else
        return get_arg_val<uint32_t>(i);
#endif
    }

    // returns false for a core without an AR role
    FORCE_INLINE bool init(uint32_t common_base, uint32_t role_base) {
        cbase = common_base;
        rbase = role_base;
        size_t a = rbase;
        role = rarg(a++);
        if (role == kRoleNone) {
            return false;
        }
        wait_value = rarg(a++);
        if constexpr (MKAR_STATS) {
            cb_reserve_back(MKAR_CB_SCALER, 1);
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(MKAR_CB_SCALER));
            for (uint32_t i = 0; i < 512; ++i) w[i] = 0;
            volatile tt_l1_ptr uint16_t* h = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(MKAR_CB_SCALER));
            for (uint32_t f = 0; f < 4; ++f)
                for (uint32_t c = 0; c < 16; ++c) h[(f << 8) + c] = 0x3980;  // 2^-12 (exact in bf16)
            cb_push_back(MKAR_CB_SCALER, 1);
        }
        sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(carg(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore));
        in_addr = carg(1);
#ifdef MKAR_SEM_BASE
        base = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_SEM_BASE);
#endif
        if (role == kRoleOwner) {
            my_chip = rarg(a++);
            num_senders = rarg(a++);
            sender_arg_base = a;
            a += num_senders;
            num_segs = rarg(a++);
            seg_base = a;
            after_segs = seg_base + 3 * num_segs;
            tail_base = after_segs + (MKAR_SRED ? 1 + MKAR_RING : 0);
        } else {
            tail_base = a;
        }
        return true;
    }

    FORCE_INLINE void signal_done() const {
        const uint32_t n = carg(2);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t xy = carg(3 + i);
            noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, MK_PS_SEM(MKAR_SEM_DONE)), 1);
        }
        noc_async_atomic_barrier();
    }

    FORCE_INLINE void deliver_x(uint32_t src, size_t a) const {
        const uint32_t xdst = rarg(a++);
        ArRect r[2];
        for (uint32_t i = 0; i < 2; ++i) {
            r[i].x0 = rarg(a++); r[i].y0 = rarg(a++); r[i].x1 = rarg(a++); r[i].y1 = rarg(a++); r[i].n = rarg(a++);
        }
        const uint32_t xflag = rarg(a++);
        MKAR_ZONE("PS-DELIVER");
        if constexpr (MKAR_ROWS > 0 && MKAR_DELIVER == 2) {
            ar_bcast(src, xdst, XCT * 2 * MKAR_ROWS * 32, r, 0xFF, noc_index);
        } else if constexpr (MKAR_ROWS > 0) {
            for (uint32_t t = 0; t < XCT; ++t) {
                ar_bcast(src + (2 * t) * MKAR_ROWS * 32, xdst + t * 2048, MKAR_ROWS * 32, r, 0xFF, noc_index);
                ar_bcast(src + (2 * t + 1) * MKAR_ROWS * 32, xdst + t * 2048 + 512, MKAR_ROWS * 32, r, 0xFF, noc_index);
            }
        } else {
            ar_bcast(src, xdst, XCT * 2048, r, 0xFF, noc_index);
        }
        noc_async_writes_flushed();
        volatile tt_l1_ptr uint32_t* f = ar_gsem(xflag);
        *f = *f + 1;
        ar_flag(xflag, r, 0xFF, noc_index);
        noc_async_write_barrier();
    }

    FORCE_INLINE void send_stats(size_t a, uint32_t it) const {
        ArRect r[2];
        size_t ra = a + 1;
        for (uint32_t i = 0; i < 2; ++i) {
            r[i].x0 = rarg(ra++); r[i].y0 = rarg(ra++); r[i].x1 = rarg(ra++); r[i].y1 = rarg(ra++); r[i].n = rarg(ra++);
        }
        a += 1 + 10 + 1;  // xdst, 2 rects, xflag
        const uint32_t rslot = rarg(a++), rtag = rarg(a++);
        cb_wait_front(MKAR_CB_PART, 1);
        MKAR_ZONE("PS-STATS");
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(MKAR_CB_PART));
        const uint32_t stage = get_write_ptr(MKAR_CB_STAGE);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
        for (uint32_t i = 0; i < 16; ++i) {
            dst[i] = src[i * 16];
            dst[16 + i] = src[512 + i * 16];
        }
        if constexpr (MKAR_ROWS > 0) {
            // fold for the consumer: row r < ROWS = sum of tile rows i = r mod ROWS, rows >= ROWS = 0 (RISC soft-float)
            volatile tt_l1_ptr float* fv = reinterpret_cast<volatile tt_l1_ptr float*>(stage);
            for (uint32_t r_ = 0; r_ < MKAR_ROWS; ++r_) {
                float acc = fv[r_];
                for (uint32_t i = r_ + MKAR_ROWS; i < 32; i += MKAR_ROWS) {
                    acc += fv[i];
                }
                fv[r_] = acc;
            }
            for (uint32_t i = MKAR_ROWS; i < 32; ++i) {
                dst[i] = 0;
            }
        }
        dst[32] = it + 1;
        ar_bcast(stage, rslot, 128, r, 0xFF, noc_index);
        noc_async_writes_flushed();
        ar_bcast(stage + 128, rtag, 16, r, 0xFF, noc_index);
        noc_async_write_barrier();
        cb_pop_front(MKAR_CB_PART, 1);
    }

    // one all-reduce (index it within the pass)
#if defined(MKAR_ATYPE) && !defined(MKAR_ABIT)
#if defined(MKAR_ATYPE_PER)   // NL64: 2 NL > 64 ARs per pass; the AR type pattern repeats with period 64 (host-asserted)
#define MKAR_ABIT(it) ((static_cast<uint64_t>(MKAR_ATYPE) >> ((it) & 63u)) & 1)
#else
#define MKAR_ABIT(it) ((static_cast<uint64_t>(MKAR_ATYPE) >> it) & 1)
#endif
#endif
#if defined(MKAR_ATYPE)   // engine MK_REAL_AAR: two call sites (MLP AR, attention AR) share one copy (EXT text budget)
    __attribute__((noinline)) void iter(uint32_t it) {
#else
    FORCE_INLINE void iter(uint32_t it) {
#endif
        done_iters = it + 1;
        CircularBuffer cb_out(MKAR_CB_OUT);
        if (role != kRoleOwner) {
            PW(2, it, 0x20);
            {
                MKAR_ZONE("PS-G-AGWAIT");
                wait_ge(sem, base + wait_value * (it + 1));  // monotonic (v0.8.7), epoch base + wrap-safe (v0.9.14)
            }
            PW(2, it, 0x21);
            cb_out.push_back(MKAR_CHUNK_TILES);
            if constexpr (MKAR_DELIVER != 0) {
                cb_out.wait_front(MKAR_CHUNK_TILES);
#if defined(MKAR_DLV_NONE)   // the engine scales x (MK_AR_GAMMA) and hands it to its delivery RISC itself
#elif defined(MKAR_DLV_SIGNAL)
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_DLV_SIGNAL) = it + 1;   // another RISC delivers
#else
                deliver_x(cb_out.get_read_ptr(), tail_base);
#endif
            }
            if constexpr (MKAR_STATS) {
                // DONE only after compute has taken Sum(x^2) from this chunk: DONE releases the next RS, and the next
                // all-gather overwrites this output buffer
                send_stats(tail_base, it);
            }
            signal_done();
            cb_out.wait_front(MKAR_CHUNK_TILES);
            cb_out.pop_front(MKAR_CHUNK_TILES);
            return;
        }
        CircularBuffer cb(MKAR_CB_RED);
        if constexpr (MKAR_FUSE_RES) {
            CircularBuffer cb_res(MKAR_CB_RES);
            cb_res.push_back(MKAR_RES_TILES);
        }
        PW(2, it, 2);
        if constexpr (MKAR_PROG) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_SEM_READY)), it + 1);
        }
        PW(2, it, 3);
#if !defined(MKAR_NOOWN)   // MKAR_NOOWN: no own slot; the compute reads this device's partial in place (MKAR_CB_OWN)
        uint32_t l1_write_addr = get_write_ptr(MKAR_CB_RED) + my_chip * MKAR_CHUNK_TILES * MKAR_IN_PAGE;
#if defined(MKAR_TRID)   // engine: barrier on our own reads only (the host RISC has its own reads in flight)
        noc_async_read_set_trid(MKAR_TRID);
#endif
#if defined(MKAR_ATYPE)
        const uint32_t in_addr = (MKAR_ABIT(it)) ? get_read_ptr(MKAR_CB_AIN) : this->in_addr;
#endif
        size_t s_idx = seg_base;
        for (uint32_t s = 0; s < num_segs; s++) {
            const uint32_t xy = rarg(s_idx++);
            const uint32_t first_tile = rarg(s_idx++);
            const uint32_t tiles = rarg(s_idx++);
            if constexpr (MKAR_ROWS > 0 && MKAR_INPK) {
                noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr), l1_write_addr, 20 * MKAR_ROWS * 32);
                l1_write_addr += 20 * MKAR_ROWS * 32;
            } else if constexpr (MKAR_ROWS > 0) {
                for (uint32_t t = 0; t < tiles; t++) {
                    const uint64_t src = get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + (first_tile + t) * 2048);
                    noc_async_read(src, l1_write_addr, MKAR_ROWS * 32);
                    noc_async_read(src + 512, l1_write_addr + MKAR_ROWS * 32, MKAR_ROWS * 32);
                    l1_write_addr += 2 * MKAR_ROWS * 32;
                }
            } else {
                noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + first_tile * MKAR_IN_PAGE), l1_write_addr,
                               tiles * MKAR_IN_PAGE);
                l1_write_addr += tiles * MKAR_IN_PAGE;
            }
        }
        if constexpr (MKAR_ROWS > 0) {
            const uint32_t zend = get_write_ptr(MKAR_CB_RED) + (my_chip + 1) * MKAR_CHUNK_TILES * MKAR_IN_PAGE;
            for (uint32_t z = l1_write_addr; z < zend;) {
                const uint32_t b = zend - z < MEM_ZEROS_SIZE ? zend - z : MEM_ZEROS_SIZE;
                noc_async_read(get_noc_addr(MEM_ZEROS_BASE), z, b);
                z += b;
            }
        }
#if defined(MKAR_TRID)
        noc_async_read_set_trid(0);
        noc_async_read_barrier_with_trid(MKAR_TRID);
#else
        noc_async_read_barrier();
#endif
#endif
        if constexpr (MKAR_SRED) {
            const uint32_t flag_base = rarg(after_segs);
            MKAR_ZONE("PS-O-RSWAIT");
            for (uint32_t k = 0; k < MKAR_RING; ++k) {
                const uint32_t slot = rarg(after_segs + 1 + k);
                if (slot != my_chip) {
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_base + 16 * slot), it + 1);
                }
                cb.push_back(MKAR_CHUNK_TILES);
            }
        } else {
            {
                MKAR_ZONE("PS-O-RSWAIT");
                PW(2, it, 4);
                wait_ge(sem, base + wait_value * (it + 1));
                PW(2, it, 5);
            }
#if defined(MKAR_NOOWN)
            cb.push_back((MKAR_RING - 1) * MKAR_CHUNK_TILES);
#else
            cb.push_back(MKAR_RING * MKAR_CHUNK_TILES);
#endif
        }
        {
            MKAR_ZONE("PS-O-REDUCE");
            cb_out.wait_front(MKAR_CHUNK_TILES);
        }
        PW(2, it, 6);
        if constexpr (MKAR_AGD) {
            const uint32_t stage = carg(3 + carg(2));
            for (uint32_t i = 0; i < num_senders; i++) {
                const uint32_t sender_xy = rarg(sender_arg_base + i);
                noc_async_write(cb_out.get_read_ptr(), get_noc_addr(sender_xy >> 16, sender_xy & 0xFFFF, stage),
                                MKAR_CHUNK_TILES * 2048);
            }
            noc_async_write_barrier();
        }
        for (uint32_t i = 0; i < num_senders; i++) {
            const uint32_t sender_xy = rarg(sender_arg_base + i);
            noc_semaphore_inc(get_noc_addr(sender_xy >> 16, sender_xy & 0xFFFF, MK_PS_SEM(MKAR_SEM_AG_READY)), 1);
        }
        if constexpr (MKAR_DELIVER != 0) {
#if defined(MKAR_DLV_NONE)
#elif defined(MKAR_DLV_SIGNAL)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_DLV_SIGNAL) = it + 1;   // another RISC delivers
#else
            deliver_x(cb_out.get_read_ptr(), tail_base);
#endif
        }
        signal_done();
        if constexpr (MKAR_STATS) {
            send_stats(tail_base, it);
        }
        cb_out.pop_front(MKAR_CHUNK_TILES);
    }

    // end of the pass: reset everything this core waited on (global semaphores keep their values across programs)
    FORCE_INLINE void fini() const {
#ifdef MKAR_SEM_BASE
        // never reset the cross-device arrival semaphore (a peer may already count into the next program): advance the base
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_SEM_BASE) = base + wait_value * done_iters;
#else
        noc_semaphore_set(sem, 0);
#endif
        if (role != kRoleOwner) {
            return;
        }
#ifdef MK_PS_GSEM
        if constexpr (MKAR_PROG) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_SEM_READY)) = 0;
        }
#endif
        if constexpr (MKAR_SRED) {
            const uint32_t flag_base = rarg(after_segs);
            for (uint32_t k = 0; k < MKAR_RING; ++k) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_base + 16 * k) = 0;
            }
        }
    }
};

}  // namespace mkar
