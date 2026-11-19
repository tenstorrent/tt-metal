// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pull bench, the CONSUMER (BRISC, NoC0) on a compute core: for each assigned chunk (stager j, n) -- runtime pairs --
// poll stager j's PRODUCED >= n + 1 (a 16-B NoC read of its sync word), pull the chunk from its ring slot n mod S
// (one PB-byte NoC read L1 -> L1 into a local landing buffer), barrier, +1 on stager j's CONSUMED[n mod S].
// PULL_DIRECT_ALL (define): no stagers -- read the assigned DRAM pages straight from the interleaved tensor (the
// all-core rotating reference); the pairs are then (page, 0).
// CT: PB, S. RT: tensor addr, probe addr, probe row, nst, then nst x
// (stager NoC x, y), then count, then count x (stager index | page, n).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pull_common.hpp"

void kernel_main() {
    constexpr uint32_t PB = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr auto t_args = TensorAccessorArgs<2>();
    uint32_t a = 0;
    const uint32_t t_addr = get_arg_val<uint32_t>(a++);
    const uint32_t probe_addr = get_arg_val<uint32_t>(a++);
    const uint32_t probe_row = get_arg_val<uint32_t>(a++);
    const uint32_t land = get_write_ptr(pullb::CB_LAND);  // 2 x PB
    const uint32_t sync = get_write_ptr(pullb::CB_SYNC);
    const uint32_t ring = get_write_ptr(pullb::CB_RING);  // the stagers' ring base (same CB layout)
    const uint32_t nst = get_arg_val<uint32_t>(a++);
    const uint32_t xy0 = a;
    a += 2 * nst;
    const uint32_t count = get_arg_val<uint32_t>(a++);
    const uint32_t pairs = a;
    const auto t = TensorAccessor(t_args, t_addr, PB);
    (void)t;
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + pullb::SYNC_B);
    const uint64_t t0 = get_timestamp();
    uint64_t wait = 0;
    uint32_t polls = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t j = get_arg_val<uint32_t>(pairs + 2 * i);
        const uint32_t n = get_arg_val<uint32_t>(pairs + 2 * i + 1);
#ifdef PULL_DIRECT_ALL
        (void)n;
#ifndef PULL_ALL_DEPTH
#define PULL_ALL_DEPTH 1
#endif
        // PULL_ALL_DEPTH reads in flight per core (landing slots i mod depth), one barrier per group
        noc_async_read_page(j, t, land + (i % PULL_ALL_DEPTH) * PB);
        if ((i + 1) % PULL_ALL_DEPTH == 0 || i + 1 == count) {
            noc_async_read_barrier();
        }
#else
        const uint32_t sx = get_arg_val<uint32_t>(xy0 + 2 * j), sy = get_arg_val<uint32_t>(xy0 + 2 * j + 1);
        const uint64_t w0 = get_timestamp();
        while (true) {
            noc_async_read(get_noc_addr(sx, sy, sync + pullb::W_PRODUCED), reinterpret_cast<uint32_t>(scratch), 16);
            noc_async_read_barrier();
            ++polls;
            invalidate_l1_cache();
            if (scratch[0] >= n + 1) {
                break;
            }
        }
        wait += get_timestamp() - w0;
        noc_async_read(get_noc_addr(sx, sy, ring + (n % S) * PB), land + (i & 1) * PB, PB);
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(sx, sy, sync + pullb::w_consumed(n % S)), 1);
#endif
    }
    noc_async_atomic_barrier();
    const uint64_t t1 = get_timestamp();
    volatile tt_l1_ptr uint32_t* row = scratch + 8;
    row[0] = static_cast<uint32_t>(t0);
    row[1] = static_cast<uint32_t>(t1);
    row[2] = count * (PB / 1024);
    row[3] = static_cast<uint32_t>(wait >> 4);
    row[4] = polls;
    row[5] = 0;
    row[6] = 0;
    row[7] = pullb::MAGIC;
    const InterleavedAddrGen<true> pg{.bank_base_address = probe_addr, .page_size = 32};
    noc_async_write(reinterpret_cast<uint32_t>(row), pg.get_noc_addr(probe_row), 32);
    noc_async_write_barrier();
}
