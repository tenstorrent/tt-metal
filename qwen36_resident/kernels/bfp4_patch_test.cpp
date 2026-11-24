// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tests/bfp4_patch_probe.py: rows r0 .. r1 of n packed bfp4 tiles (DRAM, 576 B pages) re-encoded from the same rows of
// their bf16 source tiles (DRAM, 2048 B pages), written back in place. Runtime args: q addr, x addr, n, r0, r1.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "bfp4_rtn.hpp"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0), x_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2), r0 = get_arg_val<uint32_t>(3), r1 = get_arg_val<uint32_t>(4);
    const InterleavedAddrGen<true> q{.bank_base_address = q_addr, .page_size = 576};
    const InterleavedAddrGen<true> x{.bank_base_address = x_addr, .page_size = 2048};
    const uint32_t lq = get_write_ptr(0), lx = get_write_ptr(1);
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read(q.get_noc_addr(t), lq, 576);
        noc_async_read(x.get_noc_addr(t), lx, 2048);
        noc_async_read_barrier();
        for (uint32_t r = r0; r < r1; r++) {
            patch_bfp4_row(lq, r, lx, r);
        }
        noc_async_write(lq, q.get_noc_addr(t), 576);
        noc_async_write_barrier();
    }
}
