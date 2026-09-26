// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read-bandwidth probe: stream bytes from DRAM into an L1 scratch ring and discard them.
// Runs on either data-movement RISC (BRISC -> NOC0, NCRISC -> NOC1 via the kernel config).
//
// Compile-time args:
//   0 MODE          0 = interleaved pages (page p lives in bank p % num_banks at base + (p / num_banks) * page),
//                   1 = bank-contiguous (read [offset, offset + num_bytes) of one bank, as in DRAM-sharded tensors)
//   1 CHUNK_BYTES   bytes per noc_async_read (MODE 0: the aligned page size)
//   2 RING_BYTES    L1 scratch ring size; a read barrier is taken every time the ring wraps (= bytes in flight)
//   3 NUM_BANKS
//   4 SCRATCH_CB    CB index whose L1 region is the scratch ring
// Runtime args:
//   MODE 0: base_addr, start_page, num_pages
//   MODE 1: base_addr, bank_id, bank_offset, num_bytes

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t ring_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_banks = get_compile_time_arg_val(3);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(4);

    const uint32_t ring_base = get_write_ptr(scratch_cb);
    uint32_t ring_off = 0;
    const uint32_t base = get_arg_val<uint32_t>(0);

    if constexpr (mode == 0) {
        const uint32_t start_page = get_arg_val<uint32_t>(1);
        const uint32_t num_pages = get_arg_val<uint32_t>(2);
        uint32_t bank = start_page % num_banks;
        uint32_t row = start_page / num_banks;
        for (uint32_t i = 0; i < num_pages; ++i) {
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, base + row * chunk), ring_base + ring_off, chunk);
            if (++bank == num_banks) {
                bank = 0;
                ++row;
            }
            ring_off += chunk;
            if (ring_off + chunk > ring_bytes) {
                noc_async_read_barrier();
                ring_off = 0;
            }
        }
    } else {
        const uint32_t bank_id = get_arg_val<uint32_t>(1);
        const uint32_t bank_offset = get_arg_val<uint32_t>(2);
        const uint32_t num_bytes = get_arg_val<uint32_t>(3);
        const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank_id, base + bank_offset);
        for (uint32_t done = 0; done < num_bytes; done += chunk) {
            noc_async_read(bank_noc + done, ring_base + ring_off, chunk);
            ring_off += chunk;
            if (ring_off + chunk > ring_bytes) {
                noc_async_read_barrier();
                ring_off = 0;
            }
        }
    }
    noc_async_read_barrier();
}
