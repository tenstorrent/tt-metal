// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill dequant, writer (BRISC): the unit's 32 output tiles go to an interleaved [K, N]
// tensor at tile (k, oc). gu: weight column c is member c&1 of the (gate, up) pair c>>1, so it
// goes to output column c>>1 of out_addr (gate) or out_addr2 (up).
// Runtime args: out_addr, out_addr2 (gu only, else unused), u0, nu.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr2 = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);
    constexpr uint32_t ng = get_compile_time_arg_val(0);
    constexpr uint32_t out_nt = get_compile_time_arg_val(1);  // output width in tiles
    constexpr uint32_t gu = get_compile_time_arg_val(2);
    constexpr uint32_t otb = get_compile_time_arg_val(3);  // bytes per output tile
    constexpr uint32_t cb_out = 16;
    const InterleavedAddrGen<true> og0{.bank_base_address = out_addr, .page_size = otb};
    const InterleavedAddrGen<true> og1{.bank_base_address = out_addr2, .page_size = otb};

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u / ng, G = u % ng;
        const uint32_t oc = gu ? c >> 1 : c;
        const auto& og = (gu && (c & 1)) ? og1 : og0;
        for (uint32_t q = 0; q < 4; ++q) {
            cb_wait_front(cb_out, 8);
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < 8; ++t) {
                noc_async_write(l1, get_noc_addr((G * 32 + q * 8 + t) * out_nt + oc, og), otb);
                l1 += otb;
            }
            noc_async_writes_flushed();  // L1 free to reuse; acks are collected once at the end
            cb_pop_front(cb_out, 8);
        }
    }
    noc_async_write_barrier();
}
