// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// tests/bfp8_tiny_probe.py: n bf16 batch x 32 tiles (DRAM, one page of n * 512 B) into CB 0. Args: src, n.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const InterleavedAddrGen<true> s{.bank_base_address = src, .page_size = n * 512};
    cb_reserve_back(0, n);
    noc_async_read(s.get_noc_addr(0), get_write_ptr(0), n * 512);
    noc_async_read_barrier();
    cb_push_back(0, n);
}
