// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill multi-row attn-res, writer. Per unit: takes the D1 softmax weight tiles W_d [32 tokens, 32 queries] from
// compute and hands back, for every (d, q), a tile whose column 0 is W_d[:, q] (compute broadcasts it across the
// row with an exact fp32 column broadcast); then writes the nq output tiles of every column tile j.
// Runtime args: nq output addrs, n_units, then (R, j0, j1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t D1 = get_named_compile_time_arg_val("d1");
    constexpr uint32_t NQ = get_named_compile_time_arg_val("nq");
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t cb_w = 2, cb_wc = 3, cb_out = 16;

    uint32_t oa[NQ];
    for (uint32_t q = 0; q < NQ; ++q) {
        oa[q] = get_arg_val<uint32_t>(q);
    }
    const uint32_t n_units = get_arg_val<uint32_t>(NQ);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(NQ + 1 + 3 * u);
        const uint32_t j0 = get_arg_val<uint32_t>(NQ + 2 + 3 * u);
        const uint32_t j1 = get_arg_val<uint32_t>(NQ + 3 + 3 * u);
        cb_wait_front(cb_w, D1);
        cb_reserve_back(cb_wc, D1 * NQ);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_w));
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_wc));
        for (uint32_t d = 0; d < D1; ++d) {
            for (uint32_t q = 0; q < NQ; ++q) {
                volatile tt_l1_ptr uint32_t* dst = c + (d * NQ + q) * 1024;
                for (uint32_t r = 0; r < 32; ++r) {  // element (r, q) -> (r, 0); faces 0/1 rows 0-15, 2/3 rows 16-31
                    const uint32_t fr = (r >> 4) * 2, rr = (r & 15) * 16;
                    dst[fr * 256 + rr] = w[d * 1024 + (fr + (q >> 4)) * 256 + rr + (q & 15)];
                }
            }
        }
        cb_push_back(cb_wc, D1 * NQ);
        cb_pop_front(cb_w, D1);
        for (uint32_t j = j0; j < j1; ++j) {
            cb_wait_front(cb_out, NQ);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t q = 0; q < NQ; ++q) {
                const InterleavedAddrGen<true> o = {.bank_base_address = oa[q], .page_size = 4096};
                noc_async_write(src + q * 4096, o.get_noc_addr(R * WT + j), 4096);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, NQ);
        }
    }
}
