// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// META layout: see common.h (window pieces: 3 words each; out blocks: 7 words each).
// Pack-thread (TRISC2) gatherer for the sparse-row op: walks the same META phases and segments as the
// reader, waits for the reader's window load and (GS) for each phase's neighbour semaphore, takes its offset pages
// from the reader's ring (counters in the mailbox CB) and gathers its stencil slots into CB_G3 for the unpacker.
#pragma once
// ready_n (init argument): semaphore 0's value once the window is loaded: 1, or 2 with WSPLIT (reader + writer halves)
#if (NG > 2 && (defined(TRISC_PACK) || defined(TRISC_UNPACK))) || (defined(FASTMAC) && defined(TRISC_MATH))
#include "hostdev/dev_msgs.h"
#define PFENCE() asm volatile("fence" ::: "memory")
// semaphore base of this launch, from the launch message (the compute firmware has no sem_l1_base)
inline uint32_t launch_sem_base() {
    tt_l1_ptr mailboxes_t* const mbx = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
    launch_msg_t* lm = &(mbx->launch[mbx->launch_msg_rd_ptr]);
    return lm->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX] +
           lm->kernel_config.sem_offset[ProgrammableCoreType::TENSIX];
}
#endif
#if defined(KDEBUG) && (defined(TRISC_PACK) || defined(TRISC_UNPACK))
#define TCLK() (*(volatile uint32_t*)RISCV_DEBUG_REG_WALL_CLOCK_L)
#endif
#if NG > 2 && defined(TRISC_PACK)
struct PackGather {
    volatile uint32_t* mb;
    const volatile uint32_t* m;
    uint32_t win, ring, sem, consumed, R, NPH, NIN, NNB, NB;
    uint32_t p, k, nseg_p, need_start;  // current phase, segment index within it
    const volatile uint32_t* seg;
    void init(uint32_t ready_n) {
        const uint32_t sbase = launch_sem_base();
        sem = sbase;                                           // progress slots at semaphore 1 + i
        volatile uint32_t* ready = (volatile uint32_t*)sbase;  // semaphore 0: windows + mailbox ready
        mb = (volatile uint32_t*)(get_local_cb_interface(CB_MB).fifo_wr_ptr << 4);
        win = get_local_cb_interface(CB_WIN).fifo_wr_ptr << 4;
        ring = get_local_cb_interface(CB_OFF3).fifo_wr_ptr << 4;
        m = (const volatile uint32_t*)(get_local_cb_interface(CB_META).fifo_wr_ptr << 4);
        while (*ready < ready_n) {
            PFENCE();
        }
        PFENCE();
        R = m[0];
        NPH = m[1];
        NIN = m[2];
        NNB = m[3];
        NB = m[4];
        consumed = 0;
        p = 0;
        k = 0;
        nseg_p = 0;
        need_start = 1;
    }
    void start_phase() {
        while (p < NPH) {
            const uint32_t b = m[5 + 3 * R + 7 * NB + 3 * NNB + p] & 7;  // >= 8: an SpMV pass over block & 7 (RESID)
            const volatile uint32_t* bk = m + 5 + 3 * R + 7 * b;
            seg = m + bk[0];
            nseg_p = bk[1];
            k = 0;
#ifdef GS
            const uint32_t target = p ? p + POFF : 0;  // lockstep: SpMV passes wait too (see reader.cpp)
            if (target) {
                for (uint32_t i = 0; i <= NNB; ++i) {
                    volatile uint32_t* s = (volatile uint32_t*)(sem + (1 + i) * 16);
                    while (*s < target) {
                        PFENCE();
                    }
                }
                PFENCE();
            }
#endif
            if (nseg_p) {
                return;
            }
            ++p;
        }
    }
    void segment() {
        // the wait for a new phase happens here, after the previous segment was packed (else: deadlock, since the
        // phase can only end once this core's writer has emitted that segment)
        if (need_start) {
            start_phase();
            need_start = 0;
        }
        const uint32_t q0 = seg[2 * k], q1 = seg[2 * k + 1];
        const uint32_t w0 = q0 & (TPW - 1), w1 = w0 + (q1 - q0);
        for (uint32_t j = first_of(2); j < NSLOT; j = next_of(j, 2)) {
#ifdef KDEBUG
            uint32_t t0 = TCLK();
#endif
            while (mb[2] == consumed) {
                PFENCE();
            }
            PFENCE();
#ifdef KDEBUG
            mb[20] += TCLK() - t0;
            t0 = TCLK();  // waiting for offsets
#endif
            cb_reserve_back(CB_G3, 1);
            const uint32_t g = get_local_cb_interface(CB_G3).fifo_wr_ptr << 4;
#ifdef KDEBUG
            mb[21] += TCLK() - t0;
            t0 = TCLK();  // waiting for G3 space (compute behind)
#endif
            gather_tile(win, (const uint32_t*)(ring + (consumed % RING) * OPGB), (uint32_t*)g, w0, w1);
#ifdef KDEBUG
            mb[22] += TCLK() - t0;  // gathering
#endif
            PFENCE();
            cb_push_back(CB_G3, 1);
            ++consumed;
            mb[3] = consumed;
        }
        if (++k == nseg_p) {
            ++p;
            need_start = 1;
        }
    }
};
#endif

// Unpack-thread (TRISC0) gatherer: walks the same phases / segments, takes its offset pages from CB_OFF4 (normal CB,
// the reader is the producer) and gathers straight into CB_G4's current read slot, which it then unpacks itself.
// CB_G4 is a ring of 4 slots used only by this thread: a slot is rewritten 4 gathers later, long after its unpack.
#if NG > 3 && defined(TRISC_UNPACK)
#ifndef PFENCE
#define PFENCE() asm volatile("fence" ::: "memory")
#endif
struct UnpackGather {
    const volatile uint32_t* m;
    uint32_t win, sem, R, NPH, NNB, NB, p, k, nseg_p, need_start, w0, w1;
    const volatile uint32_t* seg;
    void init(uint32_t ready_n) {
        const uint32_t sbase = launch_sem_base();
        sem = sbase;
        volatile uint32_t* ready = (volatile uint32_t*)sbase;
        win = get_local_cb_interface(CB_WIN).fifo_rd_ptr << 4;
        m = (const volatile uint32_t*)(get_local_cb_interface(CB_META).fifo_rd_ptr << 4);
        while (*ready < ready_n) {
            PFENCE();
        }
        PFENCE();
        R = m[0];
        NPH = m[1];
        NNB = m[3];
        NB = m[4];
        p = 0;
        k = 0;
        nseg_p = 0;
        need_start = 1;
    }
    void start_phase() {
        while (p < NPH) {
            const uint32_t b = m[5 + 3 * R + 7 * NB + 3 * NNB + p] & 7;  // >= 8: an SpMV pass over block & 7 (RESID)
            const volatile uint32_t* bk = m + 5 + 3 * R + 7 * b;
            seg = m + bk[0];
            nseg_p = bk[1];
            k = 0;
#ifdef GS
            const uint32_t target = p ? p + POFF : 0;  // lockstep: SpMV passes wait too (see reader.cpp)
            if (target) {
                for (uint32_t i = 0; i <= NNB; ++i) {
                    volatile uint32_t* s = (volatile uint32_t*)(sem + (1 + i) * 16);
                    while (*s < target) {
                        PFENCE();
                    }
                }
                PFENCE();
            }
#endif
            if (nseg_p) {
                return;
            }
            ++p;
        }
    }
    void segment() {
        if (need_start) {
            start_phase();
            need_start = 0;
        }
        const uint32_t q0 = seg[2 * k], q1 = seg[2 * k + 1];
        w0 = q0 & (TPW - 1);
        w1 = w0 + (q1 - q0);
        if (++k == nseg_p) {
            ++p;
            need_start = 1;
        }
    }
    void gather(uint32_t j) {
#ifdef KDEBUG
        volatile uint32_t* mbk = (volatile uint32_t*)(get_local_cb_interface(CB_MB).fifo_rd_ptr << 4);
        uint32_t t0 = TCLK();
#endif
        cb_wait_front(CB_OFF4, 1);
        PFENCE();
#ifdef KDEBUG
        mbk[24] += TCLK() - t0;
        t0 = TCLK();  // waiting for offsets
#endif
        const uint32_t o = get_local_cb_interface(CB_OFF4).fifo_rd_ptr << 4;
        const uint32_t g = get_local_cb_interface(CB_G4).fifo_rd_ptr << 4;
        gather_tile(win, (const uint32_t*)o, (uint32_t*)g, w0, w1);
#ifdef KDEBUG
        mbk[25] += TCLK() - t0;  // gathering
#endif
        PFENCE();
        cb_pop_front(CB_OFF4, 1);
    }
};
#endif

// FASTMAC (math thread): walks the same phases / segments to know which 256-word faces of each tile the segment's
// rows occupy, and accumulates s += v * g over only those faces in one SFPU pass per slot.
#if defined(FASTMAC) && defined(TRISC_MATH)
#ifndef PFENCE
#define PFENCE() asm volatile("fence" ::: "memory")
#endif
struct MathSeg {
    const volatile uint32_t* m;
    uint32_t R, NPH, NNB, NB, p, k, nseg_p, f0, f1;
    const volatile uint32_t* seg;
    void init(uint32_t ready_n) {
        volatile uint32_t* ready = (volatile uint32_t*)launch_sem_base();
        // the math firmware sets up no CB interfaces: read CB_META's L1 address from the launch's CB config table
        // (word 0 of each 4-word entry = fifo address in bytes, as circular_buffer_init.h reads it)
        tt_l1_ptr mailboxes_t* const mbx = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
        launch_msg_t* lm = &(mbx->launch[mbx->launch_msg_rd_ptr]);
        const volatile uint32_t* cbt =
            (const volatile uint32_t*)(lm->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX] +
                                       lm->kernel_config.local_cb_offset);
        m = (const volatile uint32_t*)cbt[CB_META * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
        while (*ready < ready_n) {
            PFENCE();
        }
        PFENCE();
        R = m[0];
        NPH = m[1];
        NNB = m[3];
        NB = m[4];
        p = 0;
        k = 0;
        nseg_p = 0;
        load();
    }
    void load() {
        while (p < NPH) {
            const uint32_t b = m[5 + 3 * R + 7 * NB + 3 * NNB + p] & 7;
            const volatile uint32_t* bk = m + 5 + 3 * R + 7 * b;
            seg = m + bk[0];
            nseg_p = bk[1];
            k = 0;
            if (nseg_p) {
                return;
            }
            ++p;
        }
    }
    void next() {  // faces of the current segment, then advance
        const uint32_t q0 = seg[2 * k], q1 = seg[2 * k + 1];
        const uint32_t w0 = q0 & (TPW - 1), w1 = w0 + (q1 - q0);
        f0 = w0 >> 8;
        f1 = (w1 + 255) >> 8;
        if (++k == nseg_p) {
            ++p;
            load();
        }
    }
};
#endif
