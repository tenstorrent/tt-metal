// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Weight stream sender, DRAM side (NCRISC, NOC 1; stream.py): this core's bank, front to back, SLOT bytes per c_0 page.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t SLOT = get_compile_time_arg_val(0);
    constexpr uint32_t STAGE = get_compile_time_arg_val(1);     // c_0 pages
    constexpr uint32_t INFLIGHT = get_compile_time_arg_val(2);  // reads in flight (STAGE minus the writer's batch)
    // runtime args: bank, this sender's offset in a segment, slots, slots per segment, then each segment's address
    // (the stream is cut into segments: one tensor of the whole step does not fit a contiguous DRAM range)
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t off = get_arg_val<uint32_t>(1);
    const uint32_t nslots = get_arg_val<uint32_t>(2);
    const uint32_t seg_slots = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb = 0;
    const uint64_t src0 = get_noc_addr_from_bank_id<true>(bank, 0);
    // staging page k of the CB's fifo is at base + (k % STAGE) * SLOT (the fifo wraps; a page never straddles)
    const uint32_t base = get_write_ptr(cb);
    uint32_t head = 0, inflight = 0, pushed = 0;
#ifdef STREAM_STATS
    // cycles: waiting for staging room (the ring side slow), waiting for DRAM (at runtime arg STATS_ARG's address)
    uint64_t t_room = 0, t_dram = 0;
#endif
    for (uint32_t i = 0; i < nslots; ++i) {
        if (inflight == INFLIGHT) {
#ifdef STREAM_STATS
            const uint64_t t0 = get_timestamp();
            noc_async_read_barrier_with_trid(head + 1);
            t_dram += get_timestamp() - t0;
#else
            noc_async_read_barrier_with_trid(head + 1);
#endif
            cb_push_back(cb, 1);
            ++pushed;
            head = (head + 1) % INFLIGHT;
            --inflight;
        }
#ifdef STREAM_STATS
        const uint64_t t1 = get_timestamp();
        cb_reserve_back(cb, inflight + 1);
        t_room += get_timestamp() - t1;
#else
        cb_reserve_back(cb, inflight + 1);
#endif
        const uint32_t slot = (i % INFLIGHT);
        noc_async_read_set_trid(slot + 1);
#ifndef SEND_NO_READ
        const uint32_t seg = i / seg_slots;
        noc_async_read(
            src0 + get_arg_val<uint32_t>(4 + seg) + off + (i - seg * seg_slots) * SLOT,
            base + ((pushed + inflight) % STAGE) * SLOT,
            SLOT);
#endif
        ++inflight;
    }
    while (inflight) {
        noc_async_read_barrier_with_trid(head + 1);
        cb_push_back(cb, 1);
        ++pushed;
        head = (head + 1) % INFLIGHT;
        --inflight;
    }
    noc_async_read_set_trid(0);
#ifdef STREAM_STATS
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(STATS_ADDR);
    st[0] = (uint32_t)t_room;
    st[1] = (uint32_t)t_dram;
#endif
}
