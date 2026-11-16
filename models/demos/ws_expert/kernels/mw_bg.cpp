// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast wedge repro, background traffic (BRISC on NOC0, or NCRISC on NOC1), until its stop word reaches stopv (written by the
// multicaster at the end). MODE bits, each a step of the loop:
//   1 gather-like: 32 DRAM reads of 1280 B into this core, then a read barrier (the lane BRISC's token-row gather);
//   2 own-expert-like: 4 DRAM reads of 13824 B with transaction ids 1..4, each waited for (the lane BRISC's QO);
//   4 h-like: a 2 KB write to core (hx, hy), a write barrier, then a semaphore increment there (wl_writer Q2);
//   8 stream: a 13824 B write to core (bx, by), then a flush (the fabric mux's packets towards its router).
// CT: 0 MODE, 1 stop word, 2 stamps, 3 landing (64 B aligned), 4 h landing, 5 h semaphore, 6 stream landing,
//     7 source buffer, 8 DRAM pages, then TensorAccessorArgs(DRAM buffer)
// RT: 0 stopv, 1 DRAM buffer address, 2 hx, 3 hy, 4 bx, 5 by, 6 seed
#include "api/dataflow/dataflow_api.h"
#ifndef MW_GQ
#define MW_GQ 32  // gather reads in flight
#endif

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t land = get_compile_time_arg_val(3), hland = get_compile_time_arg_val(4);
    constexpr uint32_t hsem = get_compile_time_arg_val(5), bland = get_compile_time_arg_val(6);
    constexpr uint32_t src = get_compile_time_arg_val(7), NPAGES = get_compile_time_arg_val(8);
    constexpr auto dargs = TensorAccessorArgs<9>();
    constexpr uint32_t PAGE = 16384, GB = 1280, OB = 13824, D = 4, HB = 2048;
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(1));
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(2));
    const uint32_t stopv = get_arg_val<uint32_t>(0);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(1), PAGE);
    const uint32_t hx = get_arg_val<uint32_t>(2), hy = get_arg_val<uint32_t>(3);
    const uint32_t bx = get_arg_val<uint32_t>(4), by = get_arg_val<uint32_t>(5), seed = get_arg_val<uint32_t>(6);
    uint32_t n = 0;
    while (true) {
        invalidate_l1_cache();
        if (*stop >= stopv) {
            break;
        }
        if constexpr (MODE & 1) {
            for (uint32_t r = 0; r < 32; ++r) {
                noc_async_read(ds.get_noc_addr((seed + n * 37 + r * 13) % NPAGES), land + r * GB, GB);
                if ((r + 1) % MW_GQ == 0) {
                    noc_async_read_barrier();
                }
            }
            noc_async_read_barrier();
        }
        if constexpr (MODE & 2) {
            for (uint32_t k = 0; k < D; ++k) {
                noc_async_read_set_trid(1 + k);
                noc_async_read(ds.get_noc_addr((seed + n * 7 + k * 5) % NPAGES), land + 32 * GB + k * OB, OB);
            }
            noc_async_read_set_trid(0);
            for (uint32_t k = 0; k < D; ++k) {
                while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + k)) {
                }
            }
        }
        if constexpr (MODE & 4) {
            noc_async_write(src, get_noc_addr(hx, hy, hland + (n % 4) * HB), HB);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(hx, hy, hsem), 1);
        }
        if constexpr (MODE & 8) {
            noc_async_write(src, get_noc_addr(bx, by, bland + (n % 2) * OB), OB);
            noc_async_writes_flushed();
        }
        ++n;
        st[8] = n;
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
