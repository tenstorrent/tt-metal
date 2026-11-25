// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tests/bfp8_tiny_probe.py: copies n pages (page bytes each) of a tensor to a raw byte buffer, so the host can read a
// device-packed tile's bytes. Runtime args: src addr, src page bytes, dst addr, n.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0), page = get_arg_val<uint32_t>(1);
    const uint32_t dst_addr = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3);
    const InterleavedAddrGen<true> src{.bank_base_address = src_addr, .page_size = page};
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = page};
    const uint32_t l1 = get_write_ptr(0);
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read(src.get_noc_addr(t), l1, page);
        noc_async_read_barrier();
        noc_async_write(l1, dst.get_noc_addr(t), page);
        noc_async_write_barrier();
    }
}
