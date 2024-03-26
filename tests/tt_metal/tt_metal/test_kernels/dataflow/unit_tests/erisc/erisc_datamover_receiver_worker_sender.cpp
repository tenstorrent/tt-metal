// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
// #include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_pages_total = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    InterleavedAddrGen<dst_is_dram> dest_addr_generator = {
        .bank_base_address = dst_addr, .page_size = page_size};
    DPRINT << " rws: args: " <<
        "\n\tdst_addr="<<dst_addr<<
        "\n\tdst_is_dram="<<(dst_is_dram ? "T" : "F")<<
        "\n\tnum_pages_total="<<num_pages_total<<
        "\n\tpage_size="<<page_size << "\n";

    DPRINT << " rws: noc_index " << (uint32_t)noc_index << "\n";
    DPRINT << " rws: my_x[0],my_y[0] " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "\n";
    DPRINT << " rws: my_x[1],my_y[1] " << (uint32_t)my_x[1] << "," << (uint32_t)my_y[1] << "\n";
    for (uint32_t p = 0; p < num_pages_total; ++p) {
        DPRINT << "rws: cb_wait_front\n";
        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
        DPRINT << *reinterpret_cast<uint32_t*>(l1_read_addr) << "\n";
        uint64_t dst_noc_addr = get_noc_addr(p, dest_addr_generator);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        DPRINT << "rws: write barrier complete\n";
        noc_async_write_barrier();
        DPRINT << "rws: cb_pop_front\n";
        cb_pop_front(cb_id_in0, 1);
    }

    // DPRINT << "rws: DONE\n";
    // ncrisc_noc_full_sync();
    // DPRINT << "rws: DONE DONE\n";
}
