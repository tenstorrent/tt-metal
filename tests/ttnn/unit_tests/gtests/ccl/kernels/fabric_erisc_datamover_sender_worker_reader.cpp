// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_pages_to_read_total = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_edm_buffer = 1;
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    const InterleavedAddrGen<src_is_dram> source_address_generator = {
        .bank_base_address = src_addr, .page_size = page_size};

    DPRINT << "swr: args " << "\n\tsrc_addr=" << src_addr << "\n\tsrc_is_dram=" << (src_is_dram ? "T" : "F")
           << "\n\tnum_pages_to_read_total=" << num_pages_to_read_total
           << "\n\tpages_per_edm_buffer=" << pages_per_edm_buffer << "\n\tpage_size=" << page_size << "\n";

    for (uint32_t num_pages_read = 0; num_pages_read < num_pages_to_read_total;
         num_pages_read += pages_per_edm_buffer) {
        // How can I read ahead into the circular buffer so I don't have to do an async read barrier for
        // every page? I only want to block when the CB is full
        uint32_t pages_to_read = std::min<uint32_t>(pages_per_edm_buffer, num_pages_to_read_total - num_pages_read);

        DPRINT << "swr: about to reserve " << (uint32_t)pages_to_read << " pages in CB (iteration "
               << (uint32_t)num_pages_read << ")" << "\n";
        cb_reserve_back(cb_id_in0, pages_to_read);
        DPRINT << "swr: CB reserve complete, getting write ptr" << "\n";

        uint32_t local_l1_read_addr = get_write_ptr(cb_id_in0);
        DPRINT << "swr: got write ptr=" << (uint32_t)local_l1_read_addr << ", starting async reads" << "\n";

        for (uint32_t p = 0; p < pages_to_read; ++p) {
            uint64_t src_noc_addr = get_noc_addr(num_pages_read + p, source_address_generator);
            DPRINT << "swr: starting async read page " << (uint32_t)(num_pages_read + p)
                   << " from noc_addr=" << (uint32_t)src_noc_addr << " to l1_addr=" << local_l1_read_addr << "\n";
            noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
            local_l1_read_addr += page_size;
        }

        DPRINT << "swr: waiting for async read barrier (all " << (uint32_t)pages_to_read << " pages)" << "\n";
        noc_async_read_barrier();
        DPRINT << "swr: async read barrier complete, pushing to CB" << "\n";

        cb_push_back(cb_id_in0, pages_to_read);
        DPRINT << "swr: CB push complete, " << (uint32_t)pages_to_read << " pages ready for consumer" << "\n";
    }
}
