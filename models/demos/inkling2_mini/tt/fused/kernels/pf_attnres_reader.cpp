// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill multi-row attn-res, reader. A unit is (tile row R, column tiles j0..j1) of the [T, C] states; per unit:
// the D1 logits tiles (row R of each [T, 32] fp32 logits tensor), then for every column tile j the D1 normed-state
// tiles (row R, column j). All tensors DRAM interleaved, fp32 TILE. Logits d of row R is tile R * stride_d + off_d
// of its tensor ([T, 32] per call: stride 1, offset 0; a sealed state's [T, 32 M] table of all M attn-res modules:
// stride M, offset = this module's index).
// Runtime args: D1 normed addrs, D1 logits addrs, D1 strides, D1 offsets, n_units, then (R, j0, j1) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t D1 = get_named_compile_time_arg_val("d1");
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");  // column tiles per row (C / 32)
    constexpr uint32_t cb_lg = 0, cb_n = 5;

    uint32_t na[D1], la[D1], ls[D1], lo[D1];
    for (uint32_t d = 0; d < D1; ++d) {
        na[d] = get_arg_val<uint32_t>(d);
        la[d] = get_arg_val<uint32_t>(D1 + d);
        ls[d] = get_arg_val<uint32_t>(2 * D1 + d);
        lo[d] = get_arg_val<uint32_t>(3 * D1 + d);
    }
    const uint32_t n_units = get_arg_val<uint32_t>(4 * D1);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(4 * D1 + 1 + 3 * u);
        const uint32_t j0 = get_arg_val<uint32_t>(4 * D1 + 2 + 3 * u);
        const uint32_t j1 = get_arg_val<uint32_t>(4 * D1 + 3 + 3 * u);
        cb_reserve_back(cb_lg, D1);
        uint32_t dst = get_write_ptr(cb_lg);
        for (uint32_t d = 0; d < D1; ++d) {
            const InterleavedAddrGen<true> lg = {.bank_base_address = la[d], .page_size = 4096};
            noc_async_read(lg.get_noc_addr(R * ls[d] + lo[d]), dst + d * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_lg, D1);
        for (uint32_t j = j0; j < j1; ++j) {
            cb_reserve_back(cb_n, D1);
            dst = get_write_ptr(cb_n);
            for (uint32_t d = 0; d < D1; ++d) {
                const InterleavedAddrGen<true> nd = {.bank_base_address = na[d], .page_size = 4096};
                noc_async_read(nd.get_noc_addr(R * WT + j), dst + d * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb_n, D1);
        }
    }
}
