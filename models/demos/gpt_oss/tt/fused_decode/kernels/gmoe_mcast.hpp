// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The grouped MoE gate|up reader's (gmoe_gu_reader.cpp) in0 block multicast: the cores whose units start with
// the same block (a run of consecutive cores: units are spread over the cores in order) get it from the lowest of them
// (the leader), which reads it and multicasts it into the others' CB (their first block: the CB, holding exactly one
// block, is at its base on every core) with semaphore 0 set after it; the others wait on the semaphore. The leader
// waits on no one. Reading the blocks on every core at once from the sharded source contended on the NOC (~80k cycles
// per core for a gate|up block; ~36k with 1 core in 8 reading).
//
// The core grid's NOC coordinates: common args phys_arg.. (NOC x per column, GX), then NOC y per row.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace gmoe {

struct McastRole {
    bool lead = false, follow = false;
    uint32_t f0 = 0, f1 = 0;  // leader: the followers [f0, f1]
};

// core c's role in the run of cores k around it for which starts_same(k) holds (c's own block); none when alone
template <typename F>
inline McastRole mcast_role(uint32_t c, uint32_t NC, F starts_same) {
    uint32_t lo = c, hi = c;
    while (lo > 0 && starts_same(lo - 1)) {
        --lo;
    }
    while (hi + 1 < NC && starts_same(hi + 1)) {
        ++hi;
    }
    McastRole r;
    if (hi > lo) {
        r.lead = lo == c;
        r.follow = !r.lead;
        r.f0 = c + 1;
        r.f1 = hi;
    }
    return r;
}

// the leader: l1[0, bytes) to the followers' same address (row-major core indices of a GX-wide grid: up to 3
// rectangles), then their semaphore 0 set to 1
template <uint32_t GX>
inline void mcast_send(const McastRole& r, uint32_t phys_arg, uint32_t l1, uint32_t bytes) {
    const uint32_t sem_addr = get_semaphore(0);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr) = 1;
    auto rect = [&](uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1) {
        const uint32_t px0 = get_common_arg_val<uint32_t>(phys_arg + x0);
        const uint32_t px1 = get_common_arg_val<uint32_t>(phys_arg + x1);
        const uint32_t py0 = get_common_arg_val<uint32_t>(phys_arg + GX + y0);
        const uint32_t py1 = get_common_arg_val<uint32_t>(phys_arg + GX + y1);
        const uint32_t nd = (x1 - x0 + 1) * (y1 - y0 + 1);
        noc_async_write_multicast(l1, get_noc_multicast_addr(px0, py0, px1, py1, l1), bytes, nd, true);
        noc_semaphore_set_multicast(sem_addr, get_noc_multicast_addr(px0, py0, px1, py1, sem_addr), nd);
    };
    const uint32_t ya = r.f0 / GX, yb = r.f1 / GX;
    if (ya == yb) {
        rect(r.f0 % GX, r.f1 % GX, ya, ya);
    } else {
        rect(r.f0 % GX, GX - 1, ya, ya);
        if (yb > ya + 1) {
            rect(0, GX - 1, ya + 1, yb - 1);
        }
        rect(0, r.f1 % GX, yb, yb);
    }
    noc_async_write_barrier();  // (the block is overwritten only after the multicast has left)
}

// a follower: its first block (pages of cb) from the leader
inline void mcast_receive(uint32_t cb, uint32_t pages) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    cb_reserve_back(cb, pages);
    noc_semaphore_wait(sem, 1);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb, pages);
}

}  // namespace gmoe
