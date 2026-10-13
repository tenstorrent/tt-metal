// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read micro-benchmark (test_dram_read_bench.py): a weight reader that delivers the down weights too (BRISC,
// NOC0). Two streams from its bank, interleaved in proportion so that neither runs ahead: gate/up chunks into CB pages
// for bench_fwd.cpp on the NCRISC (NOC1, as today), and down chunks read into a 2-slot ring here and written on to the
// down cores. The reader's down chunks are a contiguous run of the global down-chunk order (down core by down core),
// so a down core can be fed by two readers: segment s sends COUNT_s chunks to one down core, landing at slots
// (FIRST_s + i) % DN_SLOTS. Nothing is consumed. PUSH_DN: the down chunks go into GU_CB pages too (page = the larger
// chunk) and bench_fwd2.cpp on the NCRISC forwards both streams (NOC1); this RISC only reads.
//
// CT: 0 GU_CB, 1 GU_CHUNK, 2 N_GU, 3 DN_CB, 4 DN_CHUNK, 5 DN_SLOTS, 6 MAX_SEG (<= 4), 7 PUSH_DN
// RT: 0 bank base, 1 bank id, 2 gate/up offset, 3 down offset, 4 down landing address, 5 n segments,
//     6.. per segment: down core virtual xy, count, first chunk index in that core's sequence

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t gu_cb = get_compile_time_arg_val(0);
    constexpr uint32_t gu_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t n_gu = get_compile_time_arg_val(2);
    constexpr uint32_t dn_cb = get_compile_time_arg_val(3);
    constexpr uint32_t dn_chunk = get_compile_time_arg_val(4);
    constexpr uint32_t dn_slots = get_compile_time_arg_val(5);
    constexpr uint32_t max_seg = get_compile_time_arg_val(6);
    constexpr bool push_dn = get_compile_time_arg_val(7) != 0;
    static_assert(max_seg >= 1 && max_seg <= 4, "1..4 segments");

    const uint32_t base = get_arg_val<uint32_t>(0), bank = get_arg_val<uint32_t>(1);
    const uint64_t gu_src = get_noc_addr_from_bank_id<true>(bank, base + get_arg_val<uint32_t>(2));
    const uint64_t dn_src = get_noc_addr_from_bank_id<true>(bank, base + get_arg_val<uint32_t>(3));
    const uint32_t land = get_arg_val<uint32_t>(4), n_seg = get_arg_val<uint32_t>(5);
    uint64_t dst[max_seg];
    uint32_t cnt[max_seg], first[max_seg], n_dn = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        const uint32_t xy = get_arg_val<uint32_t>(6 + 3 * s);
        dst[s] = get_noc_addr(xy >> 16, xy & 0xFFFF, land);
        cnt[s] = get_arg_val<uint32_t>(7 + 3 * s);
        first[s] = get_arg_val<uint32_t>(8 + 3 * s);
        n_dn += cnt[s];
    }
    const uint32_t ring = get_write_ptr(dn_cb);
    uint32_t g = 0, d = 0, seg = 0, in_seg = 0;
    while (g < n_gu || d < n_dn) {
        // the stream that is behind its share goes next
        if (d >= n_dn || (g < n_gu && uint64_t(g) * n_dn <= uint64_t(d) * n_gu)) {
            cb_reserve_back(gu_cb, 1);
            noc_async_read(gu_src + uint64_t(g) * gu_chunk, get_write_ptr(gu_cb), gu_chunk);
            noc_async_read_barrier();
            cb_push_back(gu_cb, 1);
            ++g;
        } else if constexpr (push_dn) {
            cb_reserve_back(gu_cb, 1);
            noc_async_read(dn_src + uint64_t(d) * dn_chunk, get_write_ptr(gu_cb), dn_chunk);
            noc_async_read_barrier();
            cb_push_back(gu_cb, 1);
            ++d;
        } else {
            const uint32_t slot = ring + (d % 2) * dn_chunk;
            noc_async_writes_flushed();  // the slot's previous chunk has left
            noc_async_read(dn_src + uint64_t(d) * dn_chunk, slot, dn_chunk);
            noc_async_read_barrier();
            noc_async_write(slot, dst[seg] + ((first[seg] + in_seg) % dn_slots) * dn_chunk, dn_chunk);
            ++d;
            if (++in_seg == cnt[seg]) {
                ++seg;
                in_seg = 0;
            }
        }
    }
    noc_async_write_barrier();
}
