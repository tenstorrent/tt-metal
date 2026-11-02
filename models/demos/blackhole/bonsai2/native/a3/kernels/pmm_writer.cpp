// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Short-prefill ternary matmul, writer: column c's MT output tiles to rows m of output column oc
// (gu: (gate, up) pair member c&1 -> out_addr / out_addr2 at column c>>1).
// Runtime args: out_addr, out_addr2, c0, nc.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr2 = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    const uint32_t nc = get_arg_val<uint32_t>(3);
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t out_nt = get_compile_time_arg_val(1);
    constexpr uint32_t gu = get_compile_time_arg_val(2);
    constexpr uint32_t otb = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = 16;
    const InterleavedAddrGen<true> og0{.bank_base_address = out_addr, .page_size = otb};
    const InterleavedAddrGen<true> og1{.bank_base_address = out_addr2, .page_size = otb};
    for (uint32_t c = c0; c < c0 + nc; ++c) {
        const uint32_t oc = gu ? c >> 1 : c;
        const auto& og = (gu && (c & 1)) ? og1 : og0;
        cb_wait_front(cb_out, MT);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t m = 0; m < MT; ++m, l1 += otb) {
            noc_async_write(l1, get_noc_addr(m * out_nt + oc, og), otb);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, MT);
    }
    noc_async_write_barrier();
}
