// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Logits tail, step 2 reader: this core's pages of the raw logits row (dense fp32, 1024 values per page) and
// the two softcap scalar tiles (1/cap, cap).
// Runtime args: raw row address, first page, scalar tensor address.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t row_page = get_named_compile_time_arg_val("row_page");
    constexpr uint32_t cb_in = 0, cb_s = 1;

    const uint32_t raw = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t scal = get_arg_val<uint32_t>(2);

    cb_reserve_back(cb_s, 2);
    cb_reserve_back(cb_in, T);
    rowio::read_pages(scal, 4096, 0, 2, get_write_ptr(cb_s));
    const InterleavedAddrGen<true> row = {.bank_base_address = raw, .page_size = row_page};
    noc_async_read(row.get_noc_addr(0, first * 4096), get_write_ptr(cb_in), T * 4096);
    noc_async_read_barrier();
    cb_push_back(cb_s, 2);
    cb_push_back(cb_in, T);
}
