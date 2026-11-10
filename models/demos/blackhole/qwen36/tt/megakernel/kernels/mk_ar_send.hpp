// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR, AR_SEND (link-sender cores) as callable phases (AR_INTERFACE.md): mkar::SendReader (NCRISC: stages the
// reduce-scatter chunks of this (link, direction) once released, and this device's reduced chunk for the all-gather) and
// mkar::SendWriter (BRISC: owns this (link, direction)'s fabric connection for the whole pass; RS unicasts + AG line
// multicasts). One iter(it) per all-reduce of the pass. Used by the standalone persistent kernels (mk_ps_reader_h.cpp /
// mk_ps_writer_h.cpp) and by the engine's AR_SEND+GDN role binaries (spec §17), which call init once, iter per AR,
// fini once at pass end.
//
// Engine contract for the hosting kernels (what other phases on these cores must NOT touch):
//   - BRISC: the FabricConnectionManager state lives in SendWriter (keep the object alive for the whole pass, file-scope
//     static); the packet-header CB (MKAR_S_CB_HDR, 4 headers) and the worker's fabric-connection L1 words (flow-control
//     / buffer-index semaphores set up by setup_fabric_connection) must not be reused; no other phase may open a fabric
//     connection on this core's (link, direction). Other phases may use BRISC/NOC0 freely between AR iterations.
//   - CBs MKAR_S_CB_RS / MKAR_S_CB_AG (staging, n_rs and n_ag chunks) and the READY / AG_READY / DONE words are AR-only.
//
// Config macros: MKAR_S_CB_RS, MKAR_S_CB_AG, MKAR_S_CB_HDR, MKAR_S_IN_PAGE, MKAR_S_OUT_PAGE, MKAR_S_CHUNK_TILES,
//   MKAR_S_BATCH, MKAR_S_IN_PPP, MKAR_S_OUT_PPP, MKAR_S_CB_RED (slot base on owners: same address everywhere),
//   MKAR_S_MY_CHIP, MKAR_S_SEM_AG_READY, MKAR_S_SEM_DONE, MKAR_S_SEM_READY, MKAR_S_SEM_GO, MKAR_S_NOUT,
//   MKAR_S_PROG (progressive release + interleaved AG), MKAR_S_ROWS, MKAR_S_INPK, MKAR_S_READYP (+ MKAR_S_READYP0..7),
//   MKAR_ALL_RT (as mk_ar_recv.hpp).
// Reader runtime: common [in_addr, out_addr, n_owners, owner xy...]; role [n_rs, n_ag, is_go_master, per RS chunk
//   (n_segs, segs x 3), per AG chunk xy]. Writer runtime: common [sem (arrival), out_addr]; role [n_rs, n_ag,
//   fwd_range, bwd_range, per RS chunk (xy, dir << 8 | hops), per AG chunk xy, fabric connection args].
#pragma once
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include <algorithm>
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

namespace mkar {

struct SendReader {
    uint32_t cbase = 0, rbase = 0, num_rs = 0, num_ag = 0, is_go_master = 0, rs_arg_words = 0;
    uint32_t in_addr = 0, out_addr = 0;

    FORCE_INLINE uint32_t carg(uint32_t i) const {
        if constexpr (MKAR_ALL_RT) {
            return get_arg_val<uint32_t>(cbase + i);
        } else {
            return get_common_arg_val<uint32_t>(i);
        }
    }
    FORCE_INLINE uint32_t rarg(uint32_t i) const { return get_arg_val<uint32_t>(rbase + i); }
    FORCE_INLINE uint32_t readyp(uint32_t i) const {
        constexpr uint32_t rp[8] = {MKAR_S_READYP0, MKAR_S_READYP1, MKAR_S_READYP2, MKAR_S_READYP3,
                                    MKAR_S_READYP4, MKAR_S_READYP5, MKAR_S_READYP6, MKAR_S_READYP7};
        return rp[i < 8 ? i : 7];
    }

    FORCE_INLINE void init(uint32_t common_base, uint32_t role_base) {
        cbase = common_base;
        rbase = role_base;
        in_addr = carg(0);
        out_addr = carg(1);
        num_rs = rarg(0);
        num_ag = rarg(1);
        is_go_master = rarg(2);
        rs_arg_words = 0;
        for (uint32_t i = 0, k = 3; i < num_rs; i++) {
            const uint32_t ns = rarg(k);
            k += 1 + 3 * ns;
            rs_arg_words += 1 + 3 * ns;
        }
    }

    FORCE_INLINE void stage_ag(uint32_t j) const {
        CircularBuffer cb_ag(MKAR_S_CB_AG);
        cb_ag.reserve_back(MKAR_S_CHUNK_TILES);
        const uint32_t xy = rarg(3 + rs_arg_words + j);
        noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, out_addr), cb_ag.get_write_ptr(),
                       MKAR_S_CHUNK_TILES * MKAR_S_OUT_PAGE);
        noc_async_read_barrier();
        cb_ag.push_back(MKAR_S_CHUNK_TILES);
    }

    FORCE_INLINE void iter(uint32_t it) const {
#if defined(MKAR_ATYPE)   // engine (MK_REAL_AAR): bit it set = an attention AR, its input is the wo output CB on the owners
        // (the CB also spans these sender cores, so its local address is the owners' address)
        const uint32_t in_addr = ((static_cast<uint64_t>(MKAR_ATYPE) >> it) & 1) ? get_read_ptr(MKAR_CB_AIN) : this->in_addr;
#endif
        CircularBuffer cb_rs(MKAR_S_CB_RS);
        auto* done_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_S_SEM_DONE));
        auto* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_S_SEM_READY));
        auto* ag_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_S_SEM_AG_READY));
        uint32_t arg_idx = 3;
        if (it > 0) {
            noc_semaphore_wait_min(done_sem, MKAR_S_NOUT * it);
        }
        if constexpr (MKAR_S_PROG) {
            if (is_go_master) {
                const uint32_t n_own = carg(2);
                for (uint32_t o = 0; o < n_own; o++) {
                    const uint32_t xy = carg(3 + o);
                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, MK_PS_SEM(MKAR_S_SEM_GO)), 1);
                }
                noc_async_atomic_barrier();
            }
        }
        uint32_t ag_staged = 0;
        for (uint32_t done = 0; done < num_rs;) {
            const uint32_t n = std::min<uint32_t>(MKAR_S_BATCH, num_rs - done);
            if constexpr (MKAR_S_PROG) {
                volatile tt_l1_ptr uint32_t* rp =
                    MKAR_S_READYP ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readyp(done)) : ready_sem;
                const uint32_t rtarget = MKAR_S_READYP ? it + 1 : it * num_rs + done + n;
                while (*rp < rtarget) {
                    if (ag_staged < num_ag && *ag_ready >= it * num_ag + ag_staged + 1) {
                        stage_ag(ag_staged++);
                    }
                }
            }
            cb_rs.reserve_back(n * MKAR_S_CHUNK_TILES);
            uint32_t l1_write_addr = cb_rs.get_write_ptr();
            const uint32_t chunk_base = l1_write_addr;
            for (uint32_t i = 0; i < n; i++) {
                if constexpr (MKAR_S_ROWS > 0) {
                    l1_write_addr = chunk_base + i * MKAR_S_CHUNK_TILES * MKAR_S_IN_PAGE;
                }
                const uint32_t num_segs = rarg(arg_idx++);
                for (uint32_t s = 0; s < num_segs; s++) {
                    const uint32_t xy = rarg(arg_idx++);
                    const uint32_t first_tile = rarg(arg_idx++);
                    const uint32_t tiles = rarg(arg_idx++);
                    if constexpr (MKAR_S_ROWS > 0 && MKAR_S_INPK) {
                        noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr), l1_write_addr, 20 * MKAR_S_ROWS * 32);
                        l1_write_addr += 20 * MKAR_S_ROWS * 32;
                    } else if constexpr (MKAR_S_ROWS > 0) {
                        for (uint32_t t = 0; t < tiles; t++) {
                            const uint64_t src = get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + (first_tile + t) * 2048);
                            noc_async_read(src, l1_write_addr, MKAR_S_ROWS * 32);
                            noc_async_read(src + 512, l1_write_addr + MKAR_S_ROWS * 32, MKAR_S_ROWS * 32);
                            l1_write_addr += 2 * MKAR_S_ROWS * 32;
                        }
                    } else {
                        noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + first_tile * MKAR_S_IN_PAGE),
                                       l1_write_addr, tiles * MKAR_S_IN_PAGE);
                        l1_write_addr += tiles * MKAR_S_IN_PAGE;
                    }
                }
            }
            if constexpr (MKAR_S_ROWS > 0) {
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t z = chunk_base + i * MKAR_S_CHUNK_TILES * MKAR_S_IN_PAGE + 20 * MKAR_S_ROWS * 32;
                    const uint32_t zend = chunk_base + (i + 1) * MKAR_S_CHUNK_TILES * MKAR_S_IN_PAGE;
                    while (z < zend) {
                        const uint32_t b = zend - z < MEM_ZEROS_SIZE ? zend - z : MEM_ZEROS_SIZE;
                        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), z, b);
                        z += b;
                    }
                }
            }
            noc_async_read_barrier();
            cb_rs.push_back(n * MKAR_S_CHUNK_TILES);
            done += n;
        }
        if constexpr (MKAR_S_PROG) {
            for (; ag_staged < num_ag; ag_staged++) {
                noc_semaphore_wait_min(ag_ready, it * num_ag + ag_staged + 1);
                stage_ag(ag_staged);
            }
        } else if (num_ag > 0) {
            noc_semaphore_wait_min(ag_ready, num_ag * (it + 1));
            for (uint32_t j = 0; j < num_ag; j++) {
                stage_ag(j);
            }
        }
    }

    // pass end: drain the device-local semaphores this core waits on, then reset them (v0.9.13)
    FORCE_INLINE void fini(uint32_t n_iters) const {
#ifdef MK_PS_GSEM
        auto* done_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_S_SEM_DONE));
        noc_semaphore_wait_min(done_sem, MKAR_S_NOUT * n_iters);
        *done_sem = 0;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_S_SEM_AG_READY)) = 0;
        if constexpr (MKAR_S_PROG && !MKAR_S_READYP) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(MKAR_S_SEM_READY)) = 0;
        }
        if constexpr (MKAR_S_PROG && MKAR_S_READYP) {
            for (uint32_t i = 0; i < num_rs && i < 8; ++i) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readyp(i)) = 0;
            }
        }
#endif
    }
};

}  // namespace mkar
