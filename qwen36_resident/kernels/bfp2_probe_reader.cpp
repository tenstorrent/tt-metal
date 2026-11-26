// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// S0-9 bfp2 probe: one DRAM page (n tiles of `page` bytes) into CB 0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1), page = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> s{.bank_base_address = src, .page_size = n * page};
    cb_reserve_back(0, n);
    noc_async_read(s.get_noc_addr(0), get_write_ptr(0), n * page);
    noc_async_read_barrier();
    cb_push_back(0, n);
}
