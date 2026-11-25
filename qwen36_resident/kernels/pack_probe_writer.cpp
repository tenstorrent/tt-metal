// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// tests/bfp8_tiny_probe.py: CB 1's n pages (page bytes each) raw into one DRAM page. Args: dst, n, page.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t dst = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1), page = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> d{.bank_base_address = dst, .page_size = n * page};
    cb_wait_front(1, n);
    noc_async_write(get_read_ptr(1), d.get_noc_addr(0), n * page);
    noc_async_write_barrier();
    cb_pop_front(1, n);
}
