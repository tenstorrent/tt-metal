// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <c_tensix_core.h>

constexpr bool mcaster = get_compile_time_arg_val(0);
constexpr uint32_t tlx = get_compile_time_arg_val(1);
constexpr uint32_t tly = get_compile_time_arg_val(2);
constexpr uint32_t mcast_end_x = get_compile_time_arg_val(3);
constexpr uint32_t mcast_end_y = get_compile_time_arg_val(4);
constexpr uint32_t num_dests = get_compile_time_arg_val(5);
constexpr uint64_t duration = (uint64_t)get_compile_time_arg_val(6) * 1000 * 1000 * 1000;
constexpr uint32_t ucast_size = get_compile_time_arg_val(7);
constexpr uint32_t mcast_size = get_compile_time_arg_val(8);
constexpr uint32_t virtual_grid_offset_x = get_compile_time_arg_val(9);
constexpr uint32_t virtual_grid_offset_y = get_compile_time_arg_val(10);
constexpr uint32_t nrands = get_compile_time_arg_val(11);
constexpr bool enable_rnd_delay = get_compile_time_arg_val(12);
constexpr uint32_t ucast_l1_addr = get_compile_time_arg_val(13);
constexpr uint32_t mcast_l1_addr = get_compile_time_arg_val(14);

inline uint32_t next_rand(tt_l1_ptr uint8_t* rnds, uint32_t& rnd_index) {
    uint32_t rnd = rnds[rnd_index];
    rnd_index = (rnd_index + 1) & (nrands - 1);
    return rnd;
}

void kernel_main() {
    uint64_t done_time = c_tensix_core::read_wall_clock() + duration;
    tt_l1_ptr uint8_t* rnds = (tt_l1_ptr uint8_t*)(get_arg_addr(0));
    uint32_t rnd_index = 0;

    uint64_t stall_time = 0;
    while (c_tensix_core::read_wall_clock() < done_time) {
        for (uint32_t count = 0; count < 1000; count++) {
            if (enable_rnd_delay) {
                // reading time here biases us to have more ~0 cycle stalls as this
                // includes the write time
                while (c_tensix_core::read_wall_clock() < stall_time);
                stall_time = c_tensix_core::read_wall_clock() + next_rand(rnds, rnd_index);
            }

            if (mcaster) {
                uint64_t dst_noc_multicast_addr =
                    (NOC_INDEX == 0) ? get_noc_multicast_addr(tlx, tly, mcast_end_x, mcast_end_y, ucast_l1_addr)
                                     : get_noc_multicast_addr(mcast_end_x, mcast_end_y, tlx, tly, ucast_l1_addr);
                noc_async_write_multicast(mcast_l1_addr, dst_noc_multicast_addr, mcast_size, num_dests, false);
            } else {
                uint32_t dst_x, dst_y;
                uint8_t noc_addr = next_rand(rnds, rnd_index);
                dst_x = (noc_addr & 0xf) + virtual_grid_offset_x;
                dst_y = (noc_addr >> 4) + virtual_grid_offset_y;
                uint64_t noc_write_addr = NOC_XY_ADDR(NOC_X(dst_x), NOC_Y(dst_y), ucast_l1_addr);
                noc_async_write(ucast_l1_addr, noc_write_addr, ucast_size);
            }
        }
    }

    noc_async_write_barrier();
}
