// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert all-chip weight landing, credit relay (BRISC, no fabric connection of its own: both links of every
// direction carry weights). This chip's group cores write their freed-expert counts into this core's words c
// (c = g * GC + j). A count v of core c (group g) frees the slot of the group's expert n = v + 1, i.e. expert
// e' = g + (v + 1) G, owned by chip o = e' % NCHIP; only o's sender for core c waits on it (word me * NC + c >= v):
//   o == me: NoC inline write into the local lander's word;
//   else:    a 16 B entry {seq, o | c << 8, sender core xy, v} into the credit queue of this chip's link-0 branch
//            sender in o's first-hop direction, which sends it on as a fabric inline write between its packets.
// At the end a done entry to every queue.
// RT: 0 me, 1-4 link-0 branch core xy (x | y << 8) for EAST, WEST, NORTH, SOUTH (0: none), 5 local lander xy,
//     6.. per chip o, per link l: xy of o's sender core that covers this chip on link l (targets c % 2 == l)
// CT: 0 NC, 1 GC, 2 G, 3 NEXP, 4 icred, 5 NCHIP, 6 mesh id, 7 credit queue addr, 8 NQ
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0), GC = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2), NEXP = get_compile_time_arg_val(3);
    constexpr uint32_t icred = get_compile_time_arg_val(4), NCHIP = get_compile_time_arg_val(5);
    constexpr uint32_t mesh = get_compile_time_arg_val(6);
    constexpr uint32_t cq = get_compile_time_arg_val(7), NQ = get_compile_time_arg_val(8);
    const uint32_t me = get_arg_val<uint32_t>(0);
    uint32_t qxy[4], wp[4] = {0, 0, 0, 0};
    for (uint32_t d = 0; d < 4; ++d) {
        qxy[d] = get_arg_val<uint32_t>(1 + d);
    }
    const uint32_t lxy = get_arg_val<uint32_t>(5);
    uint32_t dir_of[NCHIP];
    for (uint32_t o = 0; o < NCHIP; ++o) {
        dir_of[o] = o == me ? 0 : (uint32_t)get_next_hop_router_direction(mesh, o);
    }
    // 16 B staging for the entries (the last rows of this core's icred copy are never used on the relay)
    volatile tt_l1_ptr uint32_t* ent = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + (NCHIP * NC - 1) * 16);
    auto put = [&](uint32_t d, uint32_t w1, uint32_t w2, uint32_t w3) {
        const uint32_t xy = qxy[d];
        ent[0] = wp[d] + 1;
        ent[1] = w1;
        ent[2] = w2;
        ent[3] = w3;
        noc_async_write((uint32_t)ent, get_noc_addr(xy & 0xFF, xy >> 8, cq + (wp[d] % NQ) * 16), 16);
        noc_async_writes_flushed();
        ++wp[d];
    };

    uint32_t fwd[NC];
    for (uint32_t c = 0; c < NC; ++c) {
        fwd[c] = 0;
    }
    uint32_t done = 0;
    while (done < NC) {
        invalidate_l1_cache();
        for (uint32_t c = 0; c < NC; ++c) {
            const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + c * 16);
            if (v <= fwd[c]) {
                continue;
            }
            const uint32_t g = c / GC;
            for (uint32_t u = fwd[c] + 1; u <= v; ++u) {  // (each count frees one later expert's slot)
                const uint32_t e2 = g + (u + 1) * G;
                if (e2 >= NEXP) {
                    continue;
                }
                const uint32_t o = e2 % NCHIP;
                if (o == me) {
                    noc_inline_dw_write(get_noc_addr(lxy & 0xFF, lxy >> 8, icred + (me * NC + c) * 16), u);
                } else {
                    put(dir_of[o], o | c << 8, get_arg_val<uint32_t>(6 + o * 2 + c % 2), u);
                }
            }
            fwd[c] = v;
            if (v >= (NEXP - g + G - 1) / G) {  // this core's last expert
                ++done;
            }
        }
    }
    for (uint32_t d = 0; d < 4; ++d) {
        if (qxy[d]) {
            put(d, 0xFFFFFFFF, 0, 0);
        }
    }
    noc_async_full_barrier();
}
