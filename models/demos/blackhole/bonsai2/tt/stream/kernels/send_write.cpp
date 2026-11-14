// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Weight stream sender, receiver side (BRISC, NOC 0; stream.py): slot i of this sender's stream goes to receiver
// sched[i] of its R: into that receiver's ring (S slots of SLOT bytes, persistent L1), then its "filled" counter is
// bumped. Receiver r gives slots back by bumping freed[r] on this core. Counters only grow; a slot index is count % S.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t SLOT = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t BATCH = get_compile_time_arg_val(3);
    constexpr uint32_t CH = get_compile_time_arg_val(4);  // schedule entries per schedule page
    constexpr uint32_t STAGE = get_compile_time_arg_val(5);
    constexpr auto s_args = TensorAccessorArgs<6>();
    // runtime args: slots, ring, filled, freed array, sent array, schedule tensor, first schedule page, R x (x, y)
    const uint32_t nslots = get_arg_val<uint32_t>(0);
    const uint32_t ring = get_arg_val<uint32_t>(1);
    const uint32_t filled = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    volatile tt_l1_ptr uint32_t* sent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    const auto sch = TensorAccessor(s_args, get_arg_val<uint32_t>(5), CH * 4);
    const uint32_t sch0 = get_arg_val<uint32_t>(6);
    uint32_t xs[R], ys[R];
    for (uint32_t r = 0; r < R; ++r) {
        xs[r] = get_arg_val<uint32_t>(7 + 2 * r);
        ys[r] = get_arg_val<uint32_t>(8 + 2 * r);
    }
    constexpr uint32_t cb = 0, cb_s = 1;
    const uint32_t sbuf = get_write_ptr(cb_s);
    volatile tt_l1_ptr uint32_t* sv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sbuf);
    uint32_t pending = 0, popped = 0;
#ifdef STREAM_STATS
    // cycles: whole kernel, waiting for a receiver's credit, waiting for the staging data (at sent array + 16 R)
    const uint64_t t_start = get_timestamp();
    uint64_t t_credit = 0, t_data = 0;
#endif
    const uint32_t base = get_read_ptr(cb);  // staging page k at base + (k % STAGE) * SLOT
    for (uint32_t i = 0; i < nslots; ++i) {
        if (i % CH == 0) {
            noc_async_read(sch.get_noc_addr(sch0 + i / CH), sbuf, CH * 4);
            noc_async_read_barrier();
        }
        const uint32_t r = sv[i % CH];
        const uint32_t s = sent[4 * r];
#ifndef SEND_NO_CREDIT
#ifdef STREAM_STATS
        const uint64_t tc = get_timestamp();
#endif
        while (s - freed[4 * r] >= S) {
            invalidate_l1_cache();
        }
#ifdef STREAM_STATS
        const uint64_t td = get_timestamp();
        t_credit += td - tc;
#endif
#endif
        cb_wait_front(cb, pending + 1);
#ifdef STREAM_STATS
        t_data += get_timestamp() - td;
#endif
#ifdef SEND_2NOC
        const uint8_t wn = i & 1;  // dynamic NOC mode: alternate slots between the NOCs
#else
        const uint8_t wn = noc_index;
#endif
#ifndef SEND_NO_WRITE
        noc_async_write(
            base + ((popped + pending) % STAGE) * SLOT,
            get_noc_addr(xs[r], ys[r], ring + (s % S) * SLOT, wn),
            SLOT,
            wn);
#endif
        noc_semaphore_inc(get_noc_addr(xs[r], ys[r], filled, wn), 1, wn);
        sent[4 * r] = s + 1;
        if (++pending == BATCH) {  // free the staging pages in batches: one flush per BATCH writes
#ifdef SEND_2NOC
            noc_async_writes_flushed(0);
            noc_async_writes_flushed(1);
#else
            noc_async_writes_flushed();
#endif
            cb_pop_front(cb, BATCH);
            popped += BATCH;
            pending = 0;
        }
    }
#ifdef SEND_2NOC
    noc_async_write_barrier(0);
    noc_async_write_barrier(1);
    noc_async_atomic_barrier(0);
    noc_async_atomic_barrier(1);
#endif
    noc_async_write_barrier();
    if (pending) {
        cb_pop_front(cb, pending);
    }
    noc_async_atomic_barrier();
#ifdef STREAM_STATS
    volatile tt_l1_ptr uint32_t* st = sent + 4 * R;
    st[0] = (uint32_t)(get_timestamp() - t_start);
    st[1] = (uint32_t)t_credit;
    st[2] = (uint32_t)t_data;
    st[3] = nslots;
#endif
}
