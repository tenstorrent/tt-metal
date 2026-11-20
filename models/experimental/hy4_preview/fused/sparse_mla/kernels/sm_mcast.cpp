// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-MLA K sender (data movement, on every reader / converter core): multicasts the slabs this core
// converted to the S-block's 8 attention cores, into their cb_k. A chunk occupies SLABS slabs of cb_k on the
// receivers, at cb_k's base + (chunk index % 2) * chunk bytes: cb_k has the same L1 address on every core (it is
// one CB over the readers and the attention cores; the readers use it for its address only) and the receivers
// reserve and publish it one chunk at a time. Slabs past the valid ones of a chunk are never converted: a
// converter sends its first slab in their place, which is finite data under the mask.
//
// Per chunk of the S-block: wait for this core's own slabs, wait until all receivers have reserved the chunk
// (`sem_ready`, counted up by them on every converter), send, wait for the writes to land, count the chunk up on
// the receivers (`sem_data`, one increment per converter).
//
// The semaphores are global semaphores (persistent L1 words, their addresses compile-time arguments), not program
// semaphores: a program semaphore is initialised by the dispatcher on every core at every launch, about 4 us of
// launch time each on these cores (tests/test_latency.py test_launch_shapes). They are zero between launches:
// whoever waits on one sets it back to zero when it has seen every increment.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "sm_dm.hpp"

void kernel_main() {
#ifdef SM_NOP
    return;
#endif
    constexpr uint32_t cb_k = get_named_compile_time_arg_val("cb_k");
    constexpr uint32_t cb_kout = get_named_compile_time_arg_val("cb_kout");
    constexpr uint32_t cb_hdr = get_named_compile_time_arg_val("cb_hdr");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t GROUPS = get_named_compile_time_arg_val("groups");
    constexpr uint32_t NCONV = get_named_compile_time_arg_val("converters");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk_rows");
    constexpr uint32_t DHt = get_named_compile_time_arg_val("dht");
    constexpr uint32_t DESTS = get_named_compile_time_arg_val("head_groups");
    constexpr uint32_t sem_ready = get_named_compile_time_arg_val("sem_ready");
    constexpr uint32_t sem_data = get_named_compile_time_arg_val("sem_data");
    constexpr uint32_t sem_err = get_named_compile_time_arg_val("sem_err");
    constexpr uint32_t TBL = get_named_compile_time_arg_val("tbl_mcast");
    constexpr uint32_t SLABS = CHUNK / sm::TILE_ROWS;
    static_assert(DESTS == 8, "the attention cores of an S-block are a 4 x 2 rectangle");

    constexpr auto sel_args = TensorAccessorArgs<0>();

    const sm::Role role =
        sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>();
    const uint32_t conv = role.slot;
    if (conv >= NCONV) {
        return;  // a core of the reader columns that holds no converter
    }
    const sm::Coords at = sm::block_coords(TBL, role.block);
    const uint32_t sel_addr = get_common_arg_val<uint32_t>(0);

    const auto sel = TensorAccessor(sel_args, sel_addr);
    uint32_t counts[sm::MAX_USERS];
    sm::read_counts<USERS>(sel, cb_hdr, counts);

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
    const uint64_t rect = get_noc_multicast_addr(at.att_x, at.att_y, at.att_x + 3, at.att_y + 1, 0);
    const uint32_t slab_bytes = DHt * get_tile_size(cb_kout);
    const uint32_t k_base = get_write_ptr(cb_k);  // nothing is ever pushed to cb_k on this core

#ifdef SM_MCAST_TIMING
    // Measurement only (the define is the L1 address of 16 words per reader core): per chunk, the clock at the first
    // slab's multicast, when the last slab has left this core, when every slab's write is acknowledged, after the
    // call of the counter multicast and when that one is acknowledged (the timing build waits for it per chunk).
    // Words: chunks | sum, max of: write issue -> left | left -> acknowledged | the atomic's call | call ->
    // acknowledged | kernel start, first issue, end.
    uint32_t tm_sum[4] = {0, 0, 0, 0}, tm_max[4] = {0, 0, 0, 0}, tm_first = 0;
    const uint32_t tm_start = get_timestamp_32b();
#endif
    uint32_t chunks = 0;
    sm::for_each_chunk<USERS, GROUPS, CHUNK>(counts, role.block, [&](uint32_t, uint32_t, uint32_t valid) {
        const uint32_t slabs = sm::ceil_div(valid, sm::TILE_ROWS);
        const uint32_t own = sm::own_slabs<NCONV>(slabs, conv);
        const uint32_t dst = k_base + (chunks % 2) * SLABS * slab_bytes;
        ++chunks;
        if (!sm::wait_min_bounded(ready, chunks * DESTS)) {
            // the receivers' I/O kernels report it in their status pages
            noc_semaphore_inc_multicast(rect | sem_err, 16u, DESTS);
        }
        // One slab at a time, as the converter delivers them: cb_kout is a FIFO of slabs, and the slabs of one
        // chunk need not be contiguous in it (they may straddle its wrap).
#ifdef SM_MCAST_TIMING
        uint32_t tm[5] = {0, 0, 0, 0, 0};
#endif
        for (uint32_t i = 0; i < own; ++i) {
            cb_wait_front(cb_kout, DHt);
            const uint32_t src = get_read_ptr(cb_kout);
#ifdef SM_MCAST_TIMING
            if (i == 0) {
                tm[0] = get_timestamp_32b();
                if (chunks == 1) {
                    tm_first = tm[0];
                }
            }
#endif
            noc_async_write_multicast(src, rect | (dst + (conv + i * NCONV) * slab_bytes), slab_bytes, DESTS);
            if (i == 0) {
                for (uint32_t slab = slabs; slab < SLABS; ++slab) {
                    // a slab nobody converted: filled by the converter it would belong to, or by converter 0
                    // when that one has no slab of this chunk, with the sender's first slab
                    const uint32_t filler = sm::own_slabs<NCONV>(slabs, slab % NCONV) ? slab % NCONV : 0;
                    if (filler == conv) {
                        noc_async_write_multicast(src, rect | (dst + slab * slab_bytes), slab_bytes, DESTS);
                    }
                }
            }
            noc_async_writes_flushed();  // the slab has left this core: the converter may reuse its memory
            cb_pop_front(cb_kout, DHt);
        }
#ifdef SM_MCAST_TIMING
        tm[1] = get_timestamp_32b();
#endif
        noc_async_write_barrier();  // every slab has landed before the receivers are told
#ifdef SM_MCAST_TIMING
        tm[2] = get_timestamp_32b();
#endif
        noc_semaphore_inc_multicast(rect | sem_data, 1u, DESTS);
#ifdef SM_MCAST_TIMING
        tm[3] = get_timestamp_32b();
        noc_async_atomic_barrier();
        tm[4] = get_timestamp_32b();
        if (own) {
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t d = tm[k + 1] - tm[k];
                tm_sum[k] += d;
                tm_max[k] = d > tm_max[k] ? d : tm_max[k];
            }
        }
#endif
    });
    noc_async_atomic_barrier();
    noc_semaphore_set(ready, 0);  // every increment has been waited for: zero again for the next launch
#ifdef SM_MCAST_TIMING
    {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SM_MCAST_TIMING);
        w[0] = chunks;
        for (uint32_t k = 0; k < 4; ++k) {
            w[1 + 2 * k] = tm_sum[k];
            w[2 + 2 * k] = tm_max[k];
        }
        w[9] = tm_start;
        w[10] = tm_first;
        w[11] = get_timestamp_32b();
    }
#endif
}
