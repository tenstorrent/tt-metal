// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention tail chain (wo_a -> wo_b) TRISC.  CT args: 0 cb_x, 1 cb_w, 2 cb_y, 3 cb_yg, 4 cb_o, 5 sbk1, 6 pn1, 7 sbw1,
// 8 nsub1, 9 sbk2, 10 pn2, 11 sbw2, 12 nsub2, 13 MT, 14 fp32, 15 kt1 (tiles per row tile in cb_x: 128, or 160 in the
// compact layout), 16 kt2, 17 first real x tile (16 in the compact layout).   mm1: in0 = x (kt1 tiles per row tile) x
// wo_a columns of this core -> cb_y;  mm2: in0 = all-gathered y (cb_yg, kt2 tiles per row tile) x wo_b columns of this
// core -> cb_o.

#include <cstdint>
#include "sk_mm.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_y = get_compile_time_arg_val(2);
    constexpr uint32_t cb_yg = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o = get_compile_time_arg_val(4);
    constexpr uint32_t sbk1 = get_compile_time_arg_val(5);
    constexpr uint32_t pn1 = get_compile_time_arg_val(6);
    constexpr uint32_t sbw1 = get_compile_time_arg_val(7);
    constexpr uint32_t nsub1 = get_compile_time_arg_val(8);
    constexpr uint32_t sbk2 = get_compile_time_arg_val(9);
    constexpr uint32_t pn2 = get_compile_time_arg_val(10);
    constexpr uint32_t sbw2 = get_compile_time_arg_val(11);
    constexpr uint32_t nsub2 = get_compile_time_arg_val(12);
    constexpr uint32_t MT = get_compile_time_arg_val(13);
    constexpr bool fp32 = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t kt1 = get_compile_time_arg_val(15);
    constexpr uint32_t kt2 = get_compile_time_arg_val(16);
    constexpr uint32_t a_off = get_compile_time_arg_val(17);
    constexpr uint32_t CTD =
        get_compile_time_arg_val(18);  // ct_dim: weight columns per custom_mm call (k-major pairs in DRAM)

    deepseek_compute_kernel_init();
    {
        DeviceZoneScopedN("T_MM1");
        cb_wait_front(cb_x, kt1 * MT);
        sk_mm<cb_x, cb_w, cb_y, sbk1, pn1, sbw1, nsub1, MT, fp32, CTD>(a_off, kt1);
        cb_pop_front(cb_x, kt1 * MT);
    }
    {
        DeviceZoneScopedN("T_MM2");
        cb_wait_front(cb_yg, kt2 * MT);
        sk_mm<cb_yg, cb_w, cb_o, sbk2, pn2, sbw2, nsub2, MT, fp32, CTD>(0, kt2);
        cb_pop_front(cb_yg, kt2 * MT);
    }
}
