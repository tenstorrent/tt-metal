// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender core, reader: this core's share of every tensor's pages from DRAM
// into the data buffer, one fabric packet's worth at a time, in the order
// the writer streams them. With bank packing a packet holds `ppp`
// consecutive pages of one DRAM bank (pages head, head + banks, ...), which
// are contiguous in that bank on both chips, so the writer needs one fabric
// write per bank instead of one per page.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_data = get_compile_time_arg_val(0);
constexpr uint32_t num_banks = get_compile_time_arg_val(1);
constexpr uint32_t num_tensors = get_compile_time_arg_val(2);

void kernel_main() {
    uint32_t arg = 0;
    for (uint32_t t = 0; t < num_tensors; ++t) {
        const uint32_t base = get_arg_val<uint32_t>(arg++);
        const uint32_t page_size = get_arg_val<uint32_t>(arg++);
        const uint32_t pages = get_arg_val<uint32_t>(arg++);
        const uint32_t start = get_arg_val<uint32_t>(arg++);
        const uint32_t ppp = get_arg_val<uint32_t>(arg++);
        const uint32_t packing = get_arg_val<uint32_t>(arg++);
        const InterleavedAddrGen<true> gen{.bank_base_address = base, .page_size = page_size};
        const uint32_t end = start + pages;
        if (packing != 0u) {
            const uint32_t super_block = num_banks * ppp;
            const uint32_t region = ppp * page_size;
            for (uint32_t sb = start; sb < end; sb += super_block) {
                cb_reserve_back(cb_data, 1);
                const uint32_t l1 = get_write_ptr(cb_data);
                for (uint32_t b = 0; b < num_banks; ++b) {
                    const uint32_t head = sb + b;
                    if (head >= end) {
                        break;
                    }
                    uint32_t count = 0;
                    for (uint32_t pp = head; count < ppp && pp < end; pp += num_banks) {
                        ++count;
                    }
                    noc_async_read(gen.get_noc_addr(head), l1 + b * region, count * page_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_data, 1);
            }
        } else {
            for (uint32_t p = start; p < end; p += ppp) {
                cb_reserve_back(cb_data, 1);
                const uint32_t l1 = get_write_ptr(cb_data);
                const uint32_t n = (end - p < ppp) ? end - p : ppp;
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read(gen.get_noc_addr(p + j), l1 + j * page_size, page_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_data, 1);
            }
        }
    }
}
