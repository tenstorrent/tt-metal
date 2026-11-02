// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// pmm x multicast, sender (one core outside the worker rectangle, NOC0). Per K super-block G: read this
// block of x' (MT x 32 tiles) from DRAM into its own cb_x slot, wait until every receiver has reserved
// the same slot (ready semaphore == n_recv), multicast the block to the same L1 address on the worker
// rectangle, then multicast valid = 1. Mirrors the receivers' cb_x reserve/push/pop so the slot
// addresses stay in step. Runtime args: x_addr.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// Bounded semaphore wait (debug safety): never hangs the chip; on timeout the caller reports and continues.
inline bool bounded_wait(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < (1u << 24); ++i) {
        invalidate_l1_cache();
        if (*p == v) {
            return true;
        }
    }
    return false;
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t xtb = get_compile_time_arg_val(3);
    constexpr uint32_t n_recv = get_compile_time_arg_val(4);
    constexpr uint32_t x0 = get_compile_time_arg_val(5), y0 = get_compile_time_arg_val(6);
    constexpr uint32_t x1 = get_compile_time_arg_val(7), y1 = get_compile_time_arg_val(8);
    constexpr uint32_t sem_ready_id = get_compile_time_arg_val(9), sem_valid_id = get_compile_time_arg_val(10);
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t block = MT * 32 * xtb;
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready_id));
    const uint32_t valid_addr = get_semaphore(sem_valid_id);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    *valid = 1;  // the value multicast to the receivers after each block
    const InterleavedAddrGen<true> xg{.bank_base_address = x_addr, .page_size = xtb};
    // A NOC1 multicast runs from the high corner to the low one (BRISC kernels are NOC1 on Blackhole).
    auto mc_addr = [](uint32_t a) {
        return noc_index == 1 ? get_noc_multicast_addr(x1, y1, x0, y0, a) : get_noc_multicast_addr(x0, y0, x1, y1, a);
    };
    DPRINT("XSEND start mc rect ({},{})-({},{}) n={} noc={}\n", x0, y0, x1, y1, n_recv, (uint32_t)noc_index);
    for (uint32_t G = 0; G < ng; ++G) {
        cb_reserve_back(cb_x, MT * 32);
        const uint32_t l1 = get_write_ptr(cb_x);
        uint32_t p = l1;
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t t = 0; t < 32; ++t, p += xtb) {
                noc_async_read(get_noc_addr(m * kt + G * 32 + t, xg), p, xtb);
            }
        }
        noc_async_read_barrier();
        if (!bounded_wait(ready, n_recv)) {
            DPRINT("XSEND ready timeout G={} ready={} want={}\n", G, (uint32_t)*ready, n_recv);
        }
        noc_semaphore_set(ready, 0);
        noc_async_write_multicast(l1, mc_addr(l1), block, n_recv);
        noc_async_write_barrier();
        noc_semaphore_set_multicast(valid_addr, mc_addr(valid_addr), n_recv);
        noc_async_write_barrier();
        cb_push_back(cb_x, MT * 32);
        cb_wait_front(cb_x, MT * 32);
        cb_pop_front(cb_x, MT * 32);
    }
}
