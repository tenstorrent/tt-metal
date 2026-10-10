// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Repro (NOT yet run) for noc_async_read_barrier_with_trid returning before its read has landed; see README.md.
//
// CT args: 0 L1 destination CB, 1 K (reads per batch), 2 READ_BYTES, 3 NPAGES (DRAM pages), 4 ITERS
// RT args: 0 input DRAM address (interleaved, page = READ_BYTES), 1 output DRAM address (one 64 B page)
// Defines: DRAIN_CMDBUF (drain the read command buffer before the trid barrier), GLOBAL_BARRIER
// (noc_async_read_barrier). TODO: a variant on noc_async_read_one_packet_with_state_with_trid (README question 5).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t SENTINEL = 0xDEADBEEFu;

#ifdef HAMMER
// Load generator (other cores): reads K pages per batch back to back with global barriers, ITERS x HAMMER_SCALE
// batches, no checks. Same pages as the checker (consecutive banks), so its responses share the checker's links.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t READ_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t NPAGES = get_compile_time_arg_val(3);
    constexpr uint32_t ITERS = get_compile_time_arg_val(4);
    const InterleavedAddrGen<true> in = {.bank_base_address = get_arg_val<uint32_t>(0), .page_size = READ_BYTES};
    const uint32_t l1 = get_write_ptr(cb);
    uint32_t page = get_arg_val<uint32_t>(2);  // per-core start page
    for (uint32_t it = 0; it < ITERS * HAMMER_SCALE; ++it) {
        for (uint32_t j = 0; j < K; ++j) {
            noc_async_read(get_noc_addr(page++ % NPAGES, in), l1 + j * READ_BYTES, READ_BYTES);
        }
        noc_async_read_barrier();
    }
}
#else
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t READ_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t NPAGES = get_compile_time_arg_val(3);
    constexpr uint32_t ITERS = get_compile_time_arg_val(4);
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    const InterleavedAddrGen<true> in = {.bank_base_address = in_addr, .page_size = READ_BYTES};
    const InterleavedAddrGen<true> out = {.bank_base_address = out_addr, .page_size = 64};
    const uint32_t l1 = get_write_ptr(cb);  // K x READ_BYTES

    uint32_t fails = 0, first_iter = 0xFFFFFFFF, first_j = 0xFFFFFFFF, fails_last = 0, fails_other = 0;
    uint32_t page = 0;
    for (uint32_t it = 0; it < ITERS; ++it) {
        // sentinel in the last 16 B of each destination
        for (uint32_t j = 0; j < K; ++j) {
            volatile tt_l1_ptr uint32_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + j * READ_BYTES + READ_BYTES - 16);
            tail[0] = tail[1] = tail[2] = tail[3] = SENTINEL;
        }
        const uint32_t trid = 2 + (it & 1);
        uint32_t want[K];
        noc_async_read_set_trid(trid);
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t p = page++ % NPAGES;
            want[j] = p + 1;
            noc_async_read(get_noc_addr(p, in), l1 + j * READ_BYTES, READ_BYTES);
        }
#if defined(GLOBAL_BARRIER)
        noc_async_read_barrier();
#else
#ifdef DRAIN_CMDBUF
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
#endif
        noc_async_read_barrier_with_trid(trid);
#endif
        invalidate_l1_cache();
        for (uint32_t j = 0; j < K; ++j) {
            volatile tt_l1_ptr uint32_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + j * READ_BYTES + READ_BYTES - 16);
            if (tail[3] != want[j]) {
                ++fails;
                (j + 1 == K ? fails_last : fails_other) += 1;
                if (first_iter == 0xFFFFFFFF) {
                    first_iter = it;
                    first_j = j;
                }
            }
        }
        // let every read of this batch land before the next batch rewrites the sentinels
        noc_async_read_barrier();
    }
    noc_async_read_set_trid(0);

    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);  // reuse slot 0
    res[0] = fails;
    res[1] = ITERS;
    res[2] = first_iter;
    res[3] = first_j;
    res[4] = fails_last;
    res[5] = fails_other;
    for (uint32_t i = 6; i < 16; ++i) {
        res[i] = 0;
    }
    noc_async_write(l1, get_noc_addr(0, out), 64);
    noc_async_write_barrier();
}
#endif
