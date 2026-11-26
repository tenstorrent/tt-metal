// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Verify state commit (data movement only, exact copies): n pairs (src, dst) of dense vectors of `pages` 4 KiB
// pages each (interleaved DRAM), src -> L1 -> dst. Args: n, then src0, dst0, src1, dst1, ...
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_buf = 0;
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t buf = get_write_ptr(cb_buf);
    for (uint32_t i = 0; i < n; ++i) {
        rowio::read_pages(get_arg_val<uint32_t>(1 + 2 * i), 4096, 0, T, buf);
        noc_async_read_barrier();
        rowio::write_pages(get_arg_val<uint32_t>(2 + 2 * i), 4096, 0, T, buf);
        noc_async_write_barrier();
    }
}
