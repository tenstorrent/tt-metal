// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Dataflow kernel: Tensix NOC write bandwidth benchmark to a raw NOC address.
// Used to measure Tensix → L2CPU LIM and Tensix → DRAM tile write throughput.
//
// Runtime args:
//   0: dst_noc_xy        (NOC (x,y) coordinate encoded as (y<<6)|x — upper bits of NOC addr)
//   1: dst_local_addr_lo (destination local address, low 32 bits)
//   2: dst_local_addr_hi (destination local address, high 4 bits in [35:32] — for >4 GB local addrs)
//   3: l1_src_addr       (source buffer in Tensix L1)
//   4: transfer_size     (bytes per write, must be <= L1 buffer size)
//   5: num_writes        (number of timed write iterations)
//   6: l1_result_addr    (L1 address to write benchmark results for host readback)
//
// Result layout at l1_result_addr (3 × uint64_t = 24 bytes):
//   [0]: elapsed wall-clock cycles (uint64_t)
//   [1]: total bytes transferred = num_writes * transfer_size (uint64_t)
//   [2]: sentinel 0xDEAD000100000000 | num_writes (uint64_t)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_noc_xy = get_arg_val<uint32_t>(0);
    uint32_t dst_local_addr_lo = get_arg_val<uint32_t>(1);
    uint32_t dst_local_addr_hi = get_arg_val<uint32_t>(2);
    uint32_t l1_src_addr = get_arg_val<uint32_t>(3);
    uint32_t transfer_size = get_arg_val<uint32_t>(4);
    uint32_t num_writes = get_arg_val<uint32_t>(5);
    uint32_t l1_result_addr = get_arg_val<uint32_t>(6);

    // Build 64-bit NOC address: coord bits [47:36] = (y<<6)|x, local bits [35:0]
    uint64_t dst_local_addr = ((uint64_t)dst_local_addr_hi << 32) | dst_local_addr_lo;
    uint64_t dst_noc_addr = ((uint64_t)dst_noc_xy << NOC_ADDR_LOCAL_BITS) | dst_local_addr;

    // Fill source buffer with a known pattern
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_src_addr);
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        src[i] = 0xBEEF0000u + i;
    }

    // Warmup: one write+barrier before the timed region to prime the NOC
    noc_async_write(l1_src_addr, dst_noc_addr, transfer_size);
    noc_async_write_barrier();

    // No barriers in the timed region — measures issue rate only.
    // noc_async_write internally waits for NIU command-buffer slots, so this
    // converges to steady-state issue rate once the queue fills.
    // Final barrier is outside the timed region so the kernel returns cleanly.
    uint64_t t0 = get_timestamp();
    for (uint32_t i = 0; i < num_writes; i++) {
        noc_async_write<NOC_MAX_BURST_SIZE, false, true>(l1_src_addr, dst_noc_addr, transfer_size);
    }
    noc_async_write_barrier();
    uint64_t t1 = get_timestamp();

    uint64_t elapsed = t1 - t0;
    uint64_t total_bytes = (uint64_t)num_writes * transfer_size;
    uint64_t sentinel = 0xDEAD000100000000ULL | (uint64_t)num_writes;

    // Write results to L1 for host readback as 32-bit pairs (RV32 safe)
    // Layout: [elapsed_lo, elapsed_hi, bytes_lo, bytes_hi, sentinel_lo, sentinel_hi]
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_result_addr);
    result[0] = static_cast<uint32_t>(elapsed & 0xFFFFFFFFu);
    result[1] = static_cast<uint32_t>(elapsed >> 32);
    result[2] = static_cast<uint32_t>(total_bytes & 0xFFFFFFFFu);
    result[3] = static_cast<uint32_t>(total_bytes >> 32);
    result[4] = static_cast<uint32_t>(sentinel & 0xFFFFFFFFu);
    result[5] = static_cast<uint32_t>(sentinel >> 32);
}
