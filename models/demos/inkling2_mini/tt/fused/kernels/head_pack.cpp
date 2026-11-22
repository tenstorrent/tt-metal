// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Logits tail, step 1 (one core of an unembed chunk's L1 width-sharded fp32 output): row 0 of the core's
// ntiles local tiles -> contiguous raw fp32 values -> their columns of the raw logits row in DRAM.
// Runtime args: shard L1 address, raw row address, first column.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntiles = get_named_compile_time_arg_val("ntiles");
    constexpr uint32_t row_page = get_named_compile_time_arg_val("row_page");
    constexpr uint32_t cb_scr = 0;

    const uint32_t shard = get_arg_val<uint32_t>(0);
    const uint32_t raw = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);

    const uint32_t dst = get_write_ptr(cb_scr);
    for (uint32_t j = 0; j < ntiles; ++j) {  // fp32 row 0: 16 values in face 0, 16 in face 1 (+1024 B)
        noc_async_read(get_noc_addr(shard + j * 4096), dst + j * 128, 64);
        noc_async_read(get_noc_addr(shard + j * 4096 + 1024), dst + j * 128 + 64, 64);
    }
    noc_async_read_barrier();
    const InterleavedAddrGen<true> row = {.bank_base_address = raw, .page_size = row_page};
    noc_async_write(dst, row.get_noc_addr(0, col0 * 4), ntiles * 128);
    noc_async_write_barrier();
}
