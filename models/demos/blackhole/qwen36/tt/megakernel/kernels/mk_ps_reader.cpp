// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_reduce_async one_shot: link sender (local NoC) side. See create_one_shot_descriptor in the program factory.
//   1. Stage the input tiles of every remote-owned chunk of this link in the RS CB, in the writer's send order, a batch
//      of chunks per read barrier.
//   2. Wait until every owner core of this link has reduced its chunk (ag_ready semaphore), then stage those reduced
//      chunks (the owners' output shards) in the AG CB for the writer's multicast.

#include "api/dataflow/dataflow_api.h"
// Megakernel AR step 3 (oneshot_ar.py OneShotARP): persistent copy of all_reduce_async's one_shot_reader.cpp -- ITERS all-reduces
// in ONE program, fabric connections opened once. Iteration it + 1's reduce-scatter starts only after all NOUT output
// cores of this device have iteration it's reduced chunk (DONE program semaphore, as the next layer's input would).
#include "api/dataflow/circular_buffer.h"
#include <algorithm>
#include <cstdint>

// MK_PS_PROGW (debug, the hang hunt): per-RISC progress word at MK_PS_PROGW + 16 x slot = (it << 8) | step
#ifdef MK_PS_PROGW
#define PW(slot, it, step) (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>((MK_PS_PROGW) + 16 * (slot)) = ((it) << 8) | (step))
#else
#define PW(slot, it, step)
#endif
// GSEM (spec v0.9: no program semaphores in the engine): every *_sem_id compile arg is the uniform L1 address of a
// global semaphore instead of a program-semaphore id.
#ifndef MK_PS_SEM
#ifdef MK_PS_GSEM
#define MK_PS_SEM(id) (id)
#else
#define MK_PS_SEM(id) get_semaphore(id)
#endif
#endif

constexpr uint32_t cb_rs_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_ag_id = get_compile_time_arg_val(1);
constexpr uint32_t in_page_size = get_compile_time_arg_val(2);
constexpr uint32_t out_page_size = get_compile_time_arg_val(3);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(4);
constexpr uint32_t batch_chunks = get_compile_time_arg_val(5);
constexpr uint32_t ag_ready_sem_id = get_compile_time_arg_val(6);
constexpr uint32_t ITERS = get_compile_time_arg_val(7);
constexpr uint32_t done_sem_id = get_compile_time_arg_val(8);
constexpr uint32_t NOUT = get_compile_time_arg_val(9);
// progressive producer (MK_PS_DELTA_NS > 0): chunk i of this sender's per-iteration list is read only once its owner
// released it (READY count, releases arrive in list order); the GO master (one sender) starts every iteration's
// producers on the 16 owners. common args then: [in, out, n_owners, owner xy...]
constexpr bool PROG = get_compile_time_arg_val(10) != 0;
constexpr uint32_t ready_sem_id = get_compile_time_arg_val(11);
constexpr uint32_t go_sem_id = get_compile_time_arg_val(12);
// ROWS (1..16, 0 = off): row-compact AR. Each input tile contributes only rows 0..ROWS-1 = two runs of ROWS x 32 B
// (face 0 at +0, face 1 at +512), packed back to back into the chunk's staging pages (pseudo-tiles).
constexpr uint32_t ROWS = get_compile_time_arg_val(13);
// AGD: the owner already wrote its reduced chunk into this sender's AG staging buffer; only publish the CB pages
constexpr bool AGD = get_compile_time_arg_val(14) != 0;
// INPK (with ROWS): the producer (v12 owner sum, CB_XO) already holds the chunk as the packed stream at its shard base
constexpr bool INPK = get_compile_time_arg_val(15) != 0;
// READYP (MK_PS_READYP=1, the engine's form): one READY word per position of this sender's RS list (global-semaphore
// addresses in compile args 16.. 16 + 7); chunk i is read only after ITS producer released it (>= it + 1). The count
// form (one READY counted in list order) reads a chunk early when releases arrive out of list order.
constexpr bool READYP = get_compile_time_arg_val(16) != 0;

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t num_rs_chunks = get_arg_val<uint32_t>(0);
    const uint32_t num_ag_chunks = get_arg_val<uint32_t>(1);
    const uint32_t is_go_master = get_arg_val<uint32_t>(2);
    // words of the RS segment list (per chunk: num_segs, then 3 words per segment); the AG xy list follows it
    uint32_t rs_arg_words = 0;
    for (uint32_t i = 0, k = 3; i < num_rs_chunks; i++) {
        const uint32_t ns = get_arg_val<uint32_t>(k);
        k += 1 + 3 * ns;
        rs_arg_words += 1 + 3 * ns;
    }
    auto* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(ready_sem_id));
    const uint32_t readyp[8] = {get_compile_time_arg_val(17), get_compile_time_arg_val(18), get_compile_time_arg_val(19),
                                get_compile_time_arg_val(20), get_compile_time_arg_val(21), get_compile_time_arg_val(22),
                                get_compile_time_arg_val(23), get_compile_time_arg_val(24)};
    auto* done_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(done_sem_id));

    CircularBuffer cb_rs(cb_rs_id);
    CircularBuffer cb_ag(cb_ag_id);

    for (uint32_t it = 0; it < ITERS; ++it) {
        size_t arg_idx = 3;
        PW(1, it, 0x10);
        if (it > 0) {
            noc_semaphore_wait_min(done_sem, NOUT * it);
        }
        PW(1, it, 0x11);
        if constexpr (PROG) {
            if (is_go_master) {
                const uint32_t n_own = get_common_arg_val<uint32_t>(2);
                for (uint32_t o = 0; o < n_own; o++) {
                    const uint32_t xy = get_common_arg_val<uint32_t>(3 + o);
                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, MK_PS_SEM(go_sem_id)), 1);
                }
                noc_async_atomic_barrier();
            }
        }
        // 1. Remote-owned chunks: per chunk, num_segs then (noc x << 16 | y, first tile in that input shard, tiles).
        //    PROG: while the next chunk is not released, stage any reduced chunk of ours that is ready (AG), so the
        //    all-gather of early chunks is not held behind the last reduce-scatter.
        auto* ag_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(ag_ready_sem_id));
        const size_t ag_idx = 3 + rs_arg_words;  // AG chunk xy list follows the RS segment list
        uint32_t ag_staged = 0;
        auto stage_ag = [&](uint32_t j) {
            cb_ag.reserve_back(chunk_tiles);
            if constexpr (!AGD) {
                const uint32_t xy = get_arg_val<uint32_t>(ag_idx + j);
                noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, out_addr), cb_ag.get_write_ptr(),
                               chunk_tiles * out_page_size);
                noc_async_read_barrier();
            }
            cb_ag.push_back(chunk_tiles);
        };
        for (uint32_t done = 0; done < num_rs_chunks;) {
            const uint32_t n = std::min(batch_chunks, num_rs_chunks - done);
            PW(1, it, 0x40 + done);
            if constexpr (PROG) {
                volatile tt_l1_ptr uint32_t* rp =
                    READYP ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readyp[done < 8 ? done : 7]) : ready_sem;
                const uint32_t rtarget = READYP ? it + 1 : it * num_rs_chunks + done + n;
                while (*rp < rtarget) {
                    if (ag_staged < num_ag_chunks && *ag_ready >= it * num_ag_chunks + ag_staged + 1) {
                        stage_ag(ag_staged++);
                    }
                }
            }
            PW(1, it, 0x50 + done);
            cb_rs.reserve_back(n * chunk_tiles);
            PW(1, it, 0x60 + done);
            uint32_t l1_write_addr = cb_rs.get_write_ptr();
            const uint32_t chunk_base = l1_write_addr;
            for (uint32_t i = 0; i < n; i++) {
                if constexpr (ROWS > 0) {
                    l1_write_addr = chunk_base + i * chunk_tiles * in_page_size;
                }
                const uint32_t num_segs = get_arg_val<uint32_t>(arg_idx++);
                for (uint32_t s = 0; s < num_segs; s++) {
                    const uint32_t xy = get_arg_val<uint32_t>(arg_idx++);
                    const uint32_t first_tile = get_arg_val<uint32_t>(arg_idx++);
                    const uint32_t tiles = get_arg_val<uint32_t>(arg_idx++);
                    if constexpr (ROWS > 0 && INPK) {
                        noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr), l1_write_addr, 20 * ROWS * 32);
                        l1_write_addr += 20 * ROWS * 32;
                    } else if constexpr (ROWS > 0) {
                        for (uint32_t t = 0; t < tiles; t++) {
                            const uint64_t src = get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + (first_tile + t) * 2048);
                            noc_async_read(src, l1_write_addr, ROWS * 32);
                            noc_async_read(src + 512, l1_write_addr + ROWS * 32, ROWS * 32);
                            l1_write_addr += 2 * ROWS * 32;
                        }
                    } else {
                        noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + first_tile * in_page_size),
                                       l1_write_addr, tiles * in_page_size);
                        l1_write_addr += tiles * in_page_size;
                    }
                }
            }
            if constexpr (ROWS > 0) {
                // zero the packed chunks' padding (pseudo-tile tail): the owners' row statistic sums whole pseudo-tiles
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t z = chunk_base + i * chunk_tiles * in_page_size + 20 * ROWS * 32;
                    const uint32_t zend = chunk_base + (i + 1) * chunk_tiles * in_page_size;
                    while (z < zend) {
                        const uint32_t b = zend - z < MEM_ZEROS_SIZE ? zend - z : MEM_ZEROS_SIZE;
                        noc_async_read(get_noc_addr(MEM_ZEROS_BASE), z, b);
                        z += b;
                    }
                }
            }
            noc_async_read_barrier();
            cb_rs.push_back(n * chunk_tiles);
            done += n;
        }

        PW(1, it, 0x80);
        // 2. This device's (remaining) reduced chunks of this link, once their owner cores are done.
        if (PROG) {
            for (; ag_staged < num_ag_chunks; ag_staged++) {
                noc_semaphore_wait_min(ag_ready, it * num_ag_chunks + ag_staged + 1);
                stage_ag(ag_staged);
            }
        } else if (num_ag_chunks > 0) {
            noc_semaphore_wait_min(ag_ready, num_ag_chunks * (it + 1));
            cb_ag.reserve_back(num_ag_chunks * chunk_tiles);
            uint32_t l1_write_addr = cb_ag.get_write_ptr();
            if constexpr (!AGD) {
                for (uint32_t j = 0; j < num_ag_chunks; j++) {
                    const uint32_t xy = get_arg_val<uint32_t>(ag_idx + j);
                    noc_async_read(get_noc_addr(xy >> 16, xy & 0xFFFF, out_addr), l1_write_addr,
                                   chunk_tiles * out_page_size);
                    l1_write_addr += chunk_tiles * out_page_size;
                }
                noc_async_read_barrier();
            }
            cb_ag.push_back(num_ag_chunks * chunk_tiles);
        }
    }
#ifdef MK_PS_GSEM
    // global semaphores keep their value across programs: drain the last iteration's DONE increments, then reset
    noc_semaphore_wait_min(done_sem, NOUT * ITERS);
    *done_sem = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MK_PS_SEM(ag_ready_sem_id)) = 0;
    if constexpr (PROG && !READYP) {
        *ready_sem = 0;
    }
    if constexpr (PROG && READYP) {
        for (uint32_t i = 0; i < num_rs_chunks && i < 8; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(readyp[i]) = 0;  // drained above (device-local)
        }
    }
#endif
}
