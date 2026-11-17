// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head (ws_head.build_route), NCRISC (NOC1: L1 exchanges and posted DRAM writes; no DRAM reads):
//   A. Per row tile: e = exp(picks - max) from CB 2 and the picks' expert ids from CB 3: w (t, k) = e_k / sum_k e_k
//      (soft float), each token's w row ([16] fp32, K used) to w page t (posted); records t | k << 16 | e << 20.
//   Then the exchange (wsr_exchange.hpp: B, C, the fence). Sources in core order hold contiguous token ranges, so the
//   lists come out in token order.
// RT: 0 me, 1 first row tile, 2 row tiles, 3 T, 4 w, 5 idx, 6 dest, 7 counts, 8 fence addrs, 9 stamps (2: A done,
//     3: B done, 4: C done, 63: end), 10 + c: core c's NoC x | y << 8.
// CT: wsr_exchange.hpp's.
#include "wsr_exchange.hpp"

void kernel_main() {
    using namespace wsr;
    constexpr uint32_t e_cb = tt::CBIndex::c_2, id_cb = tt::CBIndex::c_3;

    const uint32_t me = get_arg_val<uint32_t>(0), r0 = get_arg_val<uint32_t>(1), nr = get_arg_val<uint32_t>(2);
    const uint32_t T = get_arg_val<uint32_t>(3);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(4), 64);
    volatile tt_l1_ptr uint32_t* stamps = p<uint32_t>(get_arg_val<uint32_t>(9));
    auto noc_at = [&](uint32_t c, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(10 + c);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };

    // ---- A
    uint32_t nrec = 0;
    for (uint32_t rt = r0; rt < r0 + nr; ++rt) {
        cb_wait_front(e_cb, 1);
        cb_wait_front(id_cb, 1);
        volatile tt_l1_ptr uint32_t* e = p<uint32_t>(get_read_ptr(e_cb));
        volatile tt_l1_ptr uint16_t* ids = p<uint16_t>(get_read_ptr(id_cb));
        for (uint32_t r = 0; r < 32 && rt * 32 + r < T; ++r) {
            const uint32_t t = rt * 32 + r;
            float v[K], sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                v[k] = __builtin_bit_cast(float, e[at(r, k)]);
                sum += v[k];
            }
            const float inv = 1.0f / sum;
            volatile tt_l1_ptr uint32_t* wr = p<uint32_t>(wst + r * 64);
            for (uint32_t k = 0; k < 16; ++k) {
                wr[k] = k < K ? __builtin_bit_cast(uint32_t, v[k] * inv) : 0;
            }
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(wst + r * 64, ws.get_noc_addr(t), 64);
            for (uint32_t k = 0; k < K; ++k) {
                p<uint32_t>(rec)[nrec++] = t | k << 16 | static_cast<uint32_t>(ids[r * K + k]) << 20;
            }
        }
        noc_async_posted_writes_flushed();  // (the w staging rows are rewritten for the next row tile)
        cb_pop_front(e_cb, 1);
        cb_pop_front(id_cb, 1);
    }
    stamps[2] = wall();
    exchange(
        me,
        nrec,
        T,
        noc_at,
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        stamps);
}
