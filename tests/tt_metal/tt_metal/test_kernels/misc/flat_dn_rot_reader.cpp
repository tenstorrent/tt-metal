// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tensix flat-expert down-weight reader (test_drisc_flat_bench.cpp), as flat_routed_expert's se6_dw.cpp with the
// rotated layout: block k of this core's slice from bank (BANK0 + k) % BANKS at ADDR + (k / BANKS) * BLK_BYTES,
// BATCH blocks per read barrier, into an L1 ring of SLOTS blocks (never consumed). Writes the start / end wall
// clock (2 x 64 bit) at TIMING_ADDR. The gate/up blocks a Tensix reader forwards are not this kernel's job.
//
// Start barrier as drisc_flat_gu_sender.cpp (32 B slot IDX of the N-slot arrival array on the barrier core).
//
// RT: 0 DRAM bank-relative address, 1 BANK0, 2 BLK_BYTES, 3 N_BLOCKS, 4 BATCH, 5 SLOTS (multiple of BATCH), 6 ring L1
//     address, 7 timing address (32 B aligned; scratch follows at +32), 8 BANKS, 9 barrier noc_x, 10 noc_y, 11 array address,
//     12 IDX, 13 N

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t bank0 = get_arg_val<uint32_t>(1);
    const uint32_t blk = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t batch = get_arg_val<uint32_t>(4);
    const uint32_t slots = get_arg_val<uint32_t>(5);
    const uint32_t ring = get_arg_val<uint32_t>(6);
    const uint32_t timing = get_arg_val<uint32_t>(7);
    const uint32_t banks = get_arg_val<uint32_t>(8);
    {
        const uint32_t bx = get_arg_val<uint32_t>(9), by = get_arg_val<uint32_t>(10), arr = get_arg_val<uint32_t>(11);
        const uint32_t idx = get_arg_val<uint32_t>(12), nall = get_arg_val<uint32_t>(13), scratch = timing + 32;
        volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        one[0] = 1;
        noc_async_write(scratch, get_noc_addr(bx, by, arr + 32 * idx), sizeof(uint32_t));
        noc_async_write_barrier();
        volatile tt_l1_ptr uint32_t* seen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 32);
        for (bool all = false; !all;) {
            noc_async_read(get_noc_addr(bx, by, arr), scratch + 32, 32 * nall);
            noc_async_read_barrier();
            invalidate_l1_cache();
            all = true;
            for (uint32_t i = 0; i < nall && all; ++i) {
                all = seen[8 * i] != 0;
            }
        }
    }
    const uint64_t t0 = get_timestamp();
    for (uint32_t k = 0, b = 0; k < n; k += batch, ++b) {
        const uint32_t nb = n - k < batch ? n - k : batch;
        const uint32_t dst = ring + (b % (slots / batch)) * batch * blk;
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t kk = k + i;
            noc_async_read(
                get_noc_addr_from_bank_id<true>((bank0 + kk) % banks, addr + (kk / banks) * blk), dst + i * blk, blk);
        }
        noc_async_read_barrier();
    }
    const uint64_t t1 = get_timestamp();
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timing);
    out[0] = static_cast<uint32_t>(t0);
    out[1] = static_cast<uint32_t>(t0 >> 32);
    out[2] = static_cast<uint32_t>(t1);
    out[3] = static_cast<uint32_t>(t1 >> 32);
}
