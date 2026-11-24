// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pull bench, the STAGER (BRISC, NoC0) on a bank-adjacent core: its pages (p mod 8 == bank, (p / 8) mod 3 == index:
// the in-bank page list, CT-free: the host passes first page, stride 24, count) into an S-slot L1 ring; after each
// read barrier PRODUCED = n + 1; slot reuse waits CONSUMED[slot] >= n / S (the consumers' increments).
// MODE_DIRECT (define PULL_DIRECT): no ring protocol -- read the pages into the ring slots back to back (the
// bank-adjacent direct-read reference, 7a's 493 GB/s pattern).
// CT: PB (page bytes), S (slots). RT: tensor addr, first page, stride, count, probe addr, probe row.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pull_common.hpp"

void kernel_main() {
    constexpr uint32_t PB = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr auto t_args = TensorAccessorArgs<2>();
    const uint32_t t_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t probe_addr = get_arg_val<uint32_t>(4);
    const uint32_t probe_row = get_arg_val<uint32_t>(5);
    const uint32_t ring = get_write_ptr(pullb::CB_RING);  // S x PB
    const uint32_t sync = get_write_ptr(pullb::CB_SYNC);
    const auto t = TensorAccessor(t_args, t_addr, PB);
    volatile tt_l1_ptr uint32_t* produced = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + pullb::W_PRODUCED);
    *produced = 0;
    for (uint32_t k = 0; k < S; ++k) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + pullb::w_consumed(k)) = 0;
    }
    const uint64_t t0 = get_timestamp();
    uint64_t wait = 0;
    for (uint32_t n = 0; n < count; ++n) {
        const uint32_t slot = n % S;
#ifndef PULL_DIRECT
        if (n >= S) {
            volatile tt_l1_ptr uint32_t* c =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + pullb::w_consumed(slot));
            const uint64_t w0 = get_timestamp();
            while (*c < n / S) {
                invalidate_l1_cache();
            }
            wait += get_timestamp() - w0;
        }
#endif
        noc_async_read_page(first + n * stride, t, ring + slot * PB);
        noc_async_read_barrier();
#ifndef PULL_DIRECT
        *produced = n + 1;
#endif
    }
#ifndef PULL_DIRECT
    // the last S chunks: every one consumed before the program may end (no consumer reads a freed ring)
    for (uint32_t k = 0; k < S && k < count; ++k) {
        const uint32_t n_last = count - 1 - k;
        volatile tt_l1_ptr uint32_t* c =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + pullb::w_consumed(n_last % S));
        while (*c < n_last / S + 1) {
            invalidate_l1_cache();
        }
    }
#endif
    const uint64_t t1 = get_timestamp();
    // probe row: [start, end, KB, wait / 16, 0, 0, 0, magic]
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + pullb::SYNC_B);
    row[0] = static_cast<uint32_t>(t0);
    row[1] = static_cast<uint32_t>(t1);
    row[2] = count * (PB / 1024);
    row[3] = static_cast<uint32_t>(wait >> 4);
    row[4] = 0;
    row[5] = 0;
    row[6] = 0;
    row[7] = pullb::MAGIC;
    const InterleavedAddrGen<true> pg{.bank_base_address = probe_addr, .page_size = 32};
    noc_async_write(reinterpret_cast<uint32_t>(row), pg.get_noc_addr(probe_row), 32);
    noc_async_write_barrier();
}
