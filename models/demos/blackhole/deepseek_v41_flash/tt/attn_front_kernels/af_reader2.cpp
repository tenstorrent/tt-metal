// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention-front POC, NCRISC of a DRAM-bank core in the SPLIT configuration (2 compute cores per bank): ONE reader
// streams the whole bank shard at full DRAM bandwidth (two readers per bank only reach ~60 %), keeps the columns it
// owns in its own ring cb_w (as af_reader.cpp) and FORWARDS the other columns over the NoC to the ring of the helper
// core of the bank (af_recv.cpp) in units of 40 tiles, with credits (helper -> bank) and data (bank -> helper)
// semaphores. wqkv (phase 1, 8 columns per bank, 160 K tiles each): columns 0..3 local, 4..7 forwarded.  wq_b (phase 2,
// 16 columns, 40 K tiles each): even columns local, odd forwarded. CT args: 0 cb_w, 1 cb_fwd, 2 bank, 3 vc, 4 w1 addr,
// 5 w2 addr, 6 ring tiles, 7 sem credit id, 8 sem data id.  RT args: helper noc x, y.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fwd = get_compile_time_arg_val(1);
    constexpr uint32_t bank = get_compile_time_arg_val(2);
    constexpr uint32_t vc = get_compile_time_arg_val(3);
    constexpr uint32_t w1_addr = get_compile_time_arg_val(4);
    constexpr uint32_t w2_addr = get_compile_time_arg_val(5);
    constexpr uint32_t RING = get_compile_time_arg_val(6);
    constexpr uint32_t sem_credit_id = get_compile_time_arg_val(7);
    constexpr uint32_t sem_data_id = get_compile_time_arg_val(8);
    constexpr uint32_t TB = 1088;    // bfp8 tile bytes
    constexpr uint32_t PAGE_T = 10;  // tiles per NoC read packet (10880 B)
    constexpr uint32_t PAGE_B = PAGE_T * TB;
    constexpr uint32_t FU = 40;  // forward unit, tiles
    constexpr uint32_t NT = 4;   // reads in flight
    constexpr uint32_t NST = 3;  // staging slots of FU tiles
    constexpr uint32_t P1_COLS = 8, P1_LB = 80, P1_KT = 160;
    constexpr uint32_t P2_COLS = 16, P2_LB = 40, P2_KT = 40;

    const uint32_t helper_x = get_arg_val<uint32_t>(0), helper_y = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_credit_id));
    const uint32_t data_sem_addr = get_semaphore(sem_data_id);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    auto& cbi = get_local_cb_interface(cb_w);
    const uint32_t ring_base = cbi.fifo_limit - cbi.fifo_size;
    const uint32_t ring_end = ring_base + RING * TB;
    const uint32_t stage_base = get_write_ptr(cb_fwd);
    // helper's ring base = ours (same CB layout on all cores of the program)
    uint32_t l1w = get_write_ptr(cb_w);

    struct Item {
        uint32_t trid;
        bool local;
        uint32_t tiles;
        uint32_t slot;
    };
    Item q[NT];
    uint32_t q_head = 0, q_cnt = 0, local_inflight = 0, fwd_inflight = 0;
    uint32_t issued = 0, fwd_issued = 0, fwd_done = 0;

    for (uint32_t ph = 0; ph < 2; ++ph) {
        const uint32_t cols = ph == 0 ? P1_COLS : P2_COLS;
        const uint32_t kt = ph == 0 ? P1_KT : P2_KT;
        const uint32_t lb = ph == 0 ? P1_LB : P2_LB;
        uint64_t base = get_noc_addr_from_bank_id<true>(bank, ph == 0 ? w1_addr : w2_addr);
        noc_async_read_one_packet_set_state<true>(base, PAGE_B, vc);
        uint32_t rd_off = 0;
        for (uint32_t n = 0; n < cols; ++n) {
            const bool local = (n & 1) == 0;
            const uint32_t unit = local ? lb : FU;
            const uint32_t nunits = kt / unit;
            for (uint32_t u = 0; u < nunits; ++u) {
                // make room in the in-flight queue (and staging slots)
                while (q_cnt == NT || (!local && fwd_inflight == NST) || (local && local_inflight == 2)) {
                    // complete the oldest
                    Item& it = q[q_head];
                    noc_async_read_barrier_with_trid(it.trid);
                    if (it.local) {
                        cb_push_back(cb_w, it.tiles);
                        local_inflight--;
                    } else {
                        // forward: wait for the credit of this unit, write, signal
                        noc_semaphore_wait_min(credit, fwd_done + 1);
                        const uint32_t dst_off = ((fwd_done * FU) % RING) * TB;
                        const uint32_t src = stage_base + it.slot * FU * TB;
                        for (uint32_t pg = 0; pg < FU / PAGE_T; ++pg) {
                            noc_async_write(
                                src + pg * PAGE_B,
                                get_noc_addr(helper_x, helper_y, ring_base + dst_off + pg * PAGE_B),
                                PAGE_B);
                        }
                        noc_semaphore_inc(get_noc_addr(helper_x, helper_y, data_sem_addr), 1);
                        fwd_done++;
                        fwd_inflight--;
                    }
                    q_head = (q_head + 1) % NT;
                    q_cnt--;
                }
                Item it;
                it.trid = (issued % NT) + 1;
                it.local = local;
                it.tiles = unit;
                uint32_t dst;
                if (local) {
                    cb_reserve_back(cb_w, unit * (local_inflight + 1));
                    dst = l1w;
                    l1w += unit * TB;
                    if (l1w >= ring_end) {
                        l1w = ring_base;
                    }
                    local_inflight++;
                    it.slot = 0;
                } else {
                    noc_async_write_barrier();  // the staging slot of 3 units ago has been sent
                    it.slot = fwd_issued % NST;
                    dst = stage_base + it.slot * FU * TB;
                    fwd_issued++;
                    fwd_inflight++;
                }
                noc_async_read_set_trid(it.trid);
                const uint32_t pages = unit / PAGE_T;
                for (uint32_t pg = 0; pg < pages; ++pg) {
                    noc_async_read_one_packet_with_state_with_trid(base, rd_off, dst, it.trid);
                    rd_off += PAGE_B;
                    dst += PAGE_B;
                }
                q[(q_head + q_cnt) % NT] = it;
                q_cnt++;
                issued++;
            }
        }
        // drain before switching the base address
        while (q_cnt > 0) {
            Item& it = q[q_head];
            noc_async_read_barrier_with_trid(it.trid);
            if (it.local) {
                cb_push_back(cb_w, it.tiles);
                local_inflight--;
            } else {
                noc_semaphore_wait_min(credit, fwd_done + 1);
                const uint32_t dst_off = ((fwd_done * FU) % RING) * TB;
                const uint32_t src = stage_base + it.slot * FU * TB;
                for (uint32_t pg = 0; pg < FU / PAGE_T; ++pg) {
                    noc_async_write(
                        src + pg * PAGE_B, get_noc_addr(helper_x, helper_y, ring_base + dst_off + pg * PAGE_B), PAGE_B);
                }
                noc_semaphore_inc(get_noc_addr(helper_x, helper_y, data_sem_addr), 1);
                fwd_done++;
                fwd_inflight--;
            }
            q_head = (q_head + 1) % NT;
            q_cnt--;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    *credit = 0;
}
