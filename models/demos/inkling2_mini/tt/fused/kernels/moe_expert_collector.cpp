// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE expert step, collector (one data-movement core outside the 72 workers).
// Waits until all 36 gate cores wrote their 512 B hidden slot into cb_hid (sem_g == 36), then multicasts the whole
// gathered hidden (36 x 8 tiny tiles) into every worker's cb_hid (same L1 address) and sets their sem_r to 1.
// Runtime args: NOC rectangle x0 y0 x1 y1 of the workers (top-left / bottom-right in NOC0 terms), num_dests.
// X_MCAST (tt/fused/moe_bank.py) first: reads row 0 of x's 72 tiles (as 1x32 tiles, as moe_bank_reader.cpp does) into
//   its cb_x, multicasts them into every core's cb_x and sets sem_x there; runtime arg 5: x address, compile args:
//   x's TensorAccessorArgs.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t sem_g_id = get_named_compile_time_arg_val("sem_g");
constexpr uint32_t sem_r_id = get_named_compile_time_arg_val("sem_r");
constexpr uint32_t n_senders = get_named_compile_time_arg_val("n_senders");  // 36
constexpr uint32_t hid_bytes = get_named_compile_time_arg_val("hid_bytes");  // 18432
#ifdef X_MCAST
constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t x_tiles = get_named_compile_time_arg_val("x_tiles");
constexpr uint32_t sem_x_id = get_named_compile_time_arg_val("sem_x");
#endif

void kernel_main() {
    const uint32_t x0 = get_arg_val<uint32_t>(0);
    const uint32_t y0 = get_arg_val<uint32_t>(1);
    const uint32_t x1 = get_arg_val<uint32_t>(2);
    const uint32_t y1 = get_arg_val<uint32_t>(3);
    const uint32_t num_dests = get_arg_val<uint32_t>(4);
    const bool far = noc_index != 0;  // NOC1 multicasts start at the bottom-right corner
#ifdef X_MCAST
    {
        constexpr auto x_args = TensorAccessorArgs<0>();
        const auto xs = TensorAccessor(x_args, get_arg_val<uint32_t>(5));
        const uint32_t xl = get_write_ptr(cb_x);
        for (uint32_t t = 0; t < x_tiles; ++t) {  // row 0 = 16 values of face 0 + 16 of face 1
            noc_async_read(xs.get_noc_addr(t, 0), xl + t * 64, 32);
            noc_async_read(xs.get_noc_addr(t, 512), xl + t * 64 + 32, 32);
        }
        noc_async_read_barrier();
        const uint64_t x_mc =
            far ? get_noc_multicast_addr(x1, y1, x0, y0, xl) : get_noc_multicast_addr(x0, y0, x1, y1, xl);
        noc_async_write_multicast(xl, x_mc, x_tiles * 64, num_dests);
        noc_async_write_barrier();
        const uint32_t sem_x_addr = get_semaphore(sem_x_id);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_x_addr) = 1;
        const uint64_t sx_mc = far ? get_noc_multicast_addr(x1, y1, x0, y0, sem_x_addr)
                                   : get_noc_multicast_addr(x0, y0, x1, y1, sem_x_addr);
        noc_semaphore_set_multicast(sem_x_addr, sx_mc, num_dests);
        noc_async_write_barrier();
    }
#endif

    const uint32_t hid_base = get_write_ptr(cb_hid);
    volatile tt_l1_ptr uint32_t* sem_g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_g_id));
    noc_semaphore_wait(sem_g, n_senders);

    const uint64_t data_mc =
        far ? get_noc_multicast_addr(x1, y1, x0, y0, hid_base) : get_noc_multicast_addr(x0, y0, x1, y1, hid_base);
    noc_async_write_multicast(hid_base, data_mc, hid_bytes, num_dests);
    noc_async_write_barrier();

    const uint32_t sem_r_addr = get_semaphore(sem_r_id);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_r_addr) = 1;
    const uint64_t sem_mc =
        far ? get_noc_multicast_addr(x1, y1, x0, y0, sem_r_addr) : get_noc_multicast_addr(x0, y0, x1, y1, sem_r_addr);
    noc_semaphore_set_multicast(sem_r_addr, sem_mc, num_dests);
    noc_async_write_barrier();
}
