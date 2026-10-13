// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read micro-benchmark (test_dram_read_bench.py): the forwarder for bench_rd2.cpp with PUSH_DN (NCRISC, NOC1).
// Takes CB pages in the reader's order (the same proportional interleave of N_GU gate/up and the down chunks, so no
// tag is needed): a gate/up page's R blocks go to the R gate/up cores (as bench_fwd.cpp), a down page goes whole to
// the current segment's down core. Pages are freed once their writes have left the core.
//
// CT: 0 CB, 1 R (<= 8), 2 BLK_BYTES, 3 N_GU, 4 LAND_SLOTS, 5 DN_CHUNK, 6 DN_SLOTS, 7 MAX_SEG (<= 4)
// RT: 0 gate/up landing address, 1.. R gate/up cores' xy, then down landing address, n segments, per segment: down
//     core xy, count, first chunk index in that core's sequence

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t n_gu = get_compile_time_arg_val(3);
    constexpr uint32_t land_slots = get_compile_time_arg_val(4);
    constexpr uint32_t dn_chunk = get_compile_time_arg_val(5);
    constexpr uint32_t dn_slots = get_compile_time_arg_val(6);
    constexpr uint32_t max_seg = get_compile_time_arg_val(7);
    static_assert(R >= 1 && R <= 8 && max_seg >= 1 && max_seg <= 4, "1..8 receivers, 1..4 segments");

    const uint32_t land = get_arg_val<uint32_t>(0);
    uint64_t recv[R];
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t xy = get_arg_val<uint32_t>(1 + j);
        recv[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, land);
    }
    const uint32_t land_dn = get_arg_val<uint32_t>(1 + R), n_seg = get_arg_val<uint32_t>(2 + R);
    uint64_t dst[max_seg];
    uint32_t cnt[max_seg], first[max_seg], n_dn = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        const uint32_t xy = get_arg_val<uint32_t>(3 + R + 3 * s);
        dst[s] = get_noc_addr(xy >> 16, xy & 0xFFFF, land_dn);
        cnt[s] = get_arg_val<uint32_t>(4 + R + 3 * s);
        first[s] = get_arg_val<uint32_t>(5 + R + 3 * s);
        n_dn += cnt[s];
    }
    uint32_t g = 0, d = 0, seg = 0, in_seg = 0;
    while (g < n_gu || d < n_dn) {
        cb_wait_front(cb, 1);
        const uint32_t src = get_read_ptr(cb);
        if (d >= n_dn || (g < n_gu && uint64_t(g) * n_dn <= uint64_t(d) * n_gu)) {
            const uint32_t off = (g % land_slots) * blk;
            for (uint32_t j = 0; j < R; ++j) {
                noc_async_write(src + j * blk, recv[j] + off, blk);
            }
            ++g;
        } else {
            noc_async_write(src, dst[seg] + ((first[seg] + in_seg) % dn_slots) * dn_chunk, dn_chunk);
            ++d;
            if (++in_seg == cnt[seg]) {
                ++seg;
                in_seg = 0;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, 1);
    }
    noc_async_write_barrier();
}
