// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Zero-fills ranges of DRAM banks from a CB of zeros (common.zero_dram: device buffers without a host image).
// Compile-time args: 0 bytes per write (the CB's size).
// Runtime args: 0 bank, 1 ranges, then per range: bank-local address, bytes.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t chunk = get_compile_time_arg_val(0);
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t ranges = get_arg_val<uint32_t>(1);
    cb_reserve_back(0, 1);
    const uint32_t zeros = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for (uint32_t i = 0; i < chunk / 4; i++) {
        z[i] = 0;
    }
    for (uint32_t r = 0; r < ranges; r++) {
        const uint32_t addr = get_arg_val<uint32_t>(2 + 2 * r);
        const uint32_t bytes = get_arg_val<uint32_t>(3 + 2 * r);
        for (uint32_t off = 0; off < bytes; off += chunk) {
            const uint32_t n = bytes - off < chunk ? bytes - off : chunk;
            noc_async_write(zeros, get_noc_addr_from_bank_id<true>(bank, addr + off), n);
        }
    }
    noc_async_write_barrier();
}
