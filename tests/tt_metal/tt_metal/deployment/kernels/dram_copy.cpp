// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "timestamp.hpp"

#define ARGS(X)                  \
    X(uint32_t, dram_start_addr) \
    X(uint32_t, dram_end_addr)   \
    X(uint32_t, transfer_size)   \
    X(uint32_t, delta_addr)      \
    X(uint32_t, buffer0)         \
    X(uint32_t, buffer1)         \
    X(uint32_t, src_bank)        \
    X(uint32_t, dst_bank)

void kernel_main() {
    ARG_INIT(ARGS);

    uint32_t buffer = buffer0;
    uint32_t buffer_next = buffer1;

    uint64_t start = timestamp();

    uint32_t curr_addr = dram_start_addr;

    uint64_t noc_src_addr = get_noc_addr_from_bank_id<true>(src_bank, curr_addr);
    noc_async_read(noc_src_addr, buffer, transfer_size);
    noc_async_read_barrier();
    curr_addr += transfer_size;

    for (; curr_addr < dram_end_addr; curr_addr += transfer_size) {
        uint64_t noc_src_addr = get_noc_addr_from_bank_id<true>(src_bank, curr_addr);
        noc_async_read(noc_src_addr, buffer_next, transfer_size);

        uint64_t noc_dst_addr = get_noc_addr_from_bank_id<true>(dst_bank, curr_addr - transfer_size, 1);
        noc_async_write(buffer, noc_dst_addr, transfer_size);

        noc_async_read_barrier();
        noc_async_write_barrier(1);
        // noc_async_full_barrier();

        std::swap(buffer, buffer_next);
    }

    uint64_t noc_dst_addr = get_noc_addr_from_bank_id<true>(dst_bank, curr_addr - transfer_size);
    noc_async_write(buffer, noc_dst_addr, transfer_size);
    noc_async_write_barrier();

    uint64_t delta = timestamp() - start;

    *(uint64_t*)delta_addr = delta;
}
