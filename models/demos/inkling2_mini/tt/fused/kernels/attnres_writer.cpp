// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused block attention residual, writer: rms(partial) -> stack row k (dense pages), outputs -> rows 0..nq-1.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t nq = get_named_compile_time_arg_val("nq");
    constexpr uint32_t k = get_named_compile_time_arg_val("k");
    constexpr uint32_t out_rows = get_named_compile_time_arg_val("out_rows");  // > nq: last output repeated
    constexpr uint32_t cb_np = 16, cb_out = 17;

    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    cb_wait_front(cb_np, T);
    rowio::write_pages(s_addr, 4096, k * T, T, get_read_ptr(cb_np));
    if constexpr (nq > 0) {
        cb_wait_front(cb_out, nq * T);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < out_rows; ++i) {
            rowio::scatter_row(out_addr, 4096, 4, ntc, i, out + (i < nq ? i : nq - 1) * T * 4096);
        }
    }
    noc_async_write_barrier();
}
