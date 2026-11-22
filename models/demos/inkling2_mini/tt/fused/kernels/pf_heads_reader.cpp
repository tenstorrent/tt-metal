// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill per-head norm, reader. Unit = (tile row R, head h). Per unit: x tiles (R, c0 + 4 h + c), c = 0..3
// (bf16, x is [T, W] with WT tile columns); (has_scale) scale tiles 4 h + c (fp32, row 0 = the column scales);
// (has_tau) tau tile R (fp32 [T, 1] column). Plus once: the reduce scaler tile (fp32).
// Runtime args: x, scale, tau, scaler, n_units, (R, h) per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_named_compile_time_arg_val("wt");
    constexpr uint32_t C0 = get_named_compile_time_arg_val("c0");
    constexpr uint32_t has_scale = get_named_compile_time_arg_val("has_scale");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t cb_x = 0, cb_scale = 1, cb_tau = 2, cb_scaler = 3;
    const uint32_t x = get_arg_val<uint32_t>(0), scale = get_arg_val<uint32_t>(1);
    const uint32_t tau = get_arg_val<uint32_t>(2), scaler = get_arg_val<uint32_t>(3);
    const uint32_t n_units = get_arg_val<uint32_t>(4);
    const InterleavedAddrGen<true> xa = {.bank_base_address = x, .page_size = 2048};
    const InterleavedAddrGen<true> sa = {.bank_base_address = scale, .page_size = 4096};
    const InterleavedAddrGen<true> ta = {.bank_base_address = tau, .page_size = 4096};
    const InterleavedAddrGen<true> ca = {.bank_base_address = scaler, .page_size = 4096};

    cb_reserve_back(cb_scaler, 1);
    noc_async_read(ca.get_noc_addr(0), get_write_ptr(cb_scaler), 4096);
    noc_async_read_barrier();
    cb_push_back(cb_scaler, 1);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t R = get_arg_val<uint32_t>(5 + 2 * u), h = get_arg_val<uint32_t>(6 + 2 * u);
        cb_reserve_back(cb_x, 4);
        const uint32_t base = R * WT + C0 + 4 * h;
        for (uint32_t c = 0; c < 4; ++c) {
            noc_async_read(xa.get_noc_addr(base + c), get_write_ptr(cb_x) + c * 2048, 2048);
        }
        if constexpr (has_scale) {
            cb_reserve_back(cb_scale, 4);
            for (uint32_t c = 0; c < 4; ++c) {
                noc_async_read(sa.get_noc_addr(4 * h + c), get_write_ptr(cb_scale) + c * 4096, 4096);
            }
        }
        if constexpr (has_tau) {
            cb_reserve_back(cb_tau, 1);
            noc_async_read(ta.get_noc_addr(R), get_write_ptr(cb_tau), 4096);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, 4);
        if constexpr (has_scale) {
            cb_push_back(cb_scale, 4);
        }
        if constexpr (has_tau) {
            cb_push_back(cb_tau, 1);
        }
    }
}
