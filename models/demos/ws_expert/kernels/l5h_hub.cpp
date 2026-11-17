// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with 5 x 5 K-split groups, h by the group hub's multicast (ws_lane5.py, L5Cfg.hmc): the BRISC
// of the group's core 0, at the rectangle's NOC0 multicast start corner, the group's only NOC0 user. Row q (in the
// chip's compute order) once its 20 h tiles are in slot q % NS of this core's h ring (flag words (slot, producer) at
// HFLAGS, written by the owners' NCRISCs, l5h_reader.cpp) and every core of the group consumed row q - NS (HCONS words,
// one a core): the slot (20 tiles) to the same slot of the 25 cores, this one included, then the slot's flag word
// (HCOLF word 5 s) = q + 1 through the same command buffer (on one NoC and virtual channel the flag lands after its
// payload; the payload unlinked, one multicast in flight, as wl_writer.cpp's mcast_with_flag). Rows multicast so far
// go to scratch word 3 (the hub's NCRISC frees its ring slots from it).
// CT: 0 NS, 1 KTD, 2 h ring, 3 HFLAGS, 4 HCONS, 5 HCOLF, 6 h tile bytes, 7 scratch, 8 counts L1, 9 table L1
// RT: 0 stamps, 1 order length, 2-5 the group's NOC0 rectangle (x0, y0, x1, y1), 6 cores in it
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NS = get_compile_time_arg_val(0), KTD = get_compile_time_arg_val(1);
    constexpr uint32_t hring = get_compile_time_arg_val(2), hflags = get_compile_time_arg_val(3);
    constexpr uint32_t hcons = get_compile_time_arg_val(4), hcolf = get_compile_time_arg_val(5);
    constexpr uint32_t htile = get_compile_time_arg_val(6), scr = get_compile_time_arg_val(7);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(8), table_l1 = get_compile_time_arg_val(9);
    constexpr uint32_t NCOL = 5, NH = 20, GC = 25;

    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    const uint32_t nexp_g = get_arg_val<uint32_t>(1);
    const uint32_t x0 = get_arg_val<uint32_t>(2), y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4), y1 = get_arg_val<uint32_t>(5);
    const uint32_t ncores = get_arg_val<uint32_t>(6);
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    do {
        invalidate_l1_cache();
    } while (counts[0] == 0);
    volatile tt_l1_ptr uint32_t* order = W(table_l1) + 16;
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += (counts[16 + (order[n] & 0xFFFF)] + 31) / 32;
    }

    const uint32_t t_start = wall();
    uint32_t busy = 0, wait_tiles = 0, wait_cons = 0;
    for (uint32_t q = 0; q < total_rows; ++q) {
        const uint32_t s = q % NS;
        uint32_t a = wall();
        for (uint32_t i = 0; i < NH; ++i) {
            while (true) {
                invalidate_l1_cache();
                if (*W(hflags + (s * NH + i) * 16) >= q + 1) {
                    break;
                }
            }
        }
        uint32_t b = wall();
        wait_tiles += b - a;
        if (q >= NS) {
            for (uint32_t k = 0; k < GC; ++k) {
                while (true) {
                    invalidate_l1_cache();
                    if (*W(hcons + k * 16) >= q - NS + 1) {
                        break;
                    }
                }
            }
        }
        a = wall();
        wait_cons += a - b;
        *W(scr + 2 * 16) = q + 1;
        const uint32_t slot = hring + s * KTD * htile;
        noc_async_write_multicast_loopback_src(
            slot, get_noc_multicast_addr(x0, y0, x1, y1, slot), KTD * htile, ncores, false);
        noc_async_write_multicast_loopback_src(
            scr + 2 * 16, get_noc_multicast_addr(x0, y0, x1, y1, hcolf + s * NCOL * 16), 4, ncores, false);
        noc_async_writes_flushed();
        *W(scr + 3 * 16) = q + 1;
        busy += wall() - a;
    }
    noc_async_write_barrier();
    st[60] = busy;
    st[61] = wait_tiles;
    st[62] = wait_cons;
    st[495] = wall() - t_start;
    st[496] = t_start;
}
