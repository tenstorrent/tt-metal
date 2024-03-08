// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
// #include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"


void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_pages_to_read_total = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    const InterleavedAddrGen<src_is_dram> source_address_generator = {
        .bank_base_address = src_addr, .page_size = page_size};

    DPRINT << "swr: args " <<
        "\n\tsrc_addr="<<src_addr<<
        "\n\tsrc_is_dram="<<(src_is_dram?"T":"F")<<
        "\n\tnum_pages_to_read_total="<<num_pages_to_read_total<<
        "\n\tpage_size="<<page_size<<"\n";

    for (uint32_t num_pages_read = 0; num_pages_read < num_pages_to_read_total; num_pages_read++) {
        // How can I read ahead into the circular buffer so I don't have to do an async read barrier for
        // every page? I only want to block when the CB is full
        DPRINT << "swr: cb_reserve_back on page " << num_pages_read << "\n";
        cb_reserve_back(cb_id_in0, 1);
        uint32_t local_l1_read_addr = get_write_ptr(cb_id_in0);
        uint64_t src_noc_addr = get_noc_addr(num_pages_read, source_address_generator);
        DPRINT << "swr: async_read\n";
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        DPRINT << "swr: read_barrier\n";
        noc_async_read_barrier();
        DPRINT << "swr: cb_push_back\n";
        cb_push_back(cb_id_in0, 1);
    }

}
