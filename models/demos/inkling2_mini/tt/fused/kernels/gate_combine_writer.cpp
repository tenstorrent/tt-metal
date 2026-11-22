// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate combine, writer: rows 0..R-1 of the core's output tiles -> row-form fp32 DRAM tensor (the
// padding rows of the zeroed output buffer are never written).
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_out = 17;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);

    const InterleavedAddrGen<true> dst = {.bank_base_address = out_addr, .page_size = 4096};
    cb_wait_front(cb_out, nt);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t j = 0; j < nt; ++j) {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t off = rowio::row_offset(r, 4);
            noc_async_write(src + j * 4096 + off, dst.get_noc_addr(col0 + j, off), 64);
            noc_async_write(src + j * 4096 + off + 1024, dst.get_noc_addr(col0 + j, off + 1024), 64);
        }
    }
    noc_async_write_barrier();
}
