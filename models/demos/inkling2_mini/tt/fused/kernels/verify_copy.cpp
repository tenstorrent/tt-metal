// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Speculative verify commit: copy whole interleaved DRAM tensors src -> dst (same page size and count), several pairs
// per core, in one program (the accepted token's conv-window state into each layer's window state).
// Runtime args: n_pairs, then (src, dst) per pair. Named: pages (per tensor), page_bytes.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t pages = get_named_compile_time_arg_val("pages");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("page_bytes");
    constexpr uint32_t batch = 8;  // pages in flight
    const uint32_t buf = get_write_ptr(0);
    const uint32_t n = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < n; ++i) {
        const InterleavedAddrGen<true> s = {
            .bank_base_address = get_arg_val<uint32_t>(1 + 2 * i), .page_size = page_bytes};
        const InterleavedAddrGen<true> d = {
            .bank_base_address = get_arg_val<uint32_t>(2 + 2 * i), .page_size = page_bytes};
        for (uint32_t p0 = 0; p0 < pages; p0 += batch) {
            const uint32_t m = pages - p0 < batch ? pages - p0 : batch;
            for (uint32_t j = 0; j < m; ++j) {
                noc_async_read(s.get_noc_addr(p0 + j), buf + j * page_bytes, page_bytes);
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < m; ++j) {
                noc_async_write(buf + j * page_bytes, d.get_noc_addr(p0 + j), page_bytes);
            }
            noc_async_write_barrier();
        }
    }
}
