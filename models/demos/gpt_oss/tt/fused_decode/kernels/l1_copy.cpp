// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One contiguous copy from this core's L1 (BRISC): bytes from src to dst, dst in this core's L1 (dst_dram = 0) or in
// DRAM bank 0 (dst_dram = 1) (the DFlash drafter context capture: a layer's residual row tiles into this device's
// slice of the context input, dflash/verify.py).
//
// runtime args: [src_addr, dst_addr, bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0);
    const uint32_t dst = get_arg_val<uint32_t>(1);
    const uint32_t bytes = get_arg_val<uint32_t>(2);
    constexpr uint32_t dst_dram = get_compile_time_arg_val(0);
    if (bytes > 0) {
        const uint64_t d = dst_dram ? get_noc_addr_from_bank_id<true>(0, dst) : get_noc_addr(dst);
        noc_async_write(src, d, bytes);
        noc_async_write_barrier();
    }
}
