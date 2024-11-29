// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr std::uint32_t iteration = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);

    std::uint32_t noc_x = get_arg_val<uint32_t>(0);
    std::uint32_t noc_y = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = 0;
    uint32_t l1_read_addr = get_read_ptr(cb_id);

    // DPRINT << noc_index <<ENDL();

    DPRINT << "Start" <<ENDL();

    // DPRINT << "risc: " << (uint)risc_type << " noc: " << (uint32_t)noc_index << " " << get_noc_nonposted_writes_acked<risc_type>(noc_index) << " " << NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) <<ENDL();
    // DPRINT << "risc: " << (uint)risc_type << " noc: " << (uint32_t)(1-noc_index) << " " << get_noc_nonposted_writes_acked<risc_type>(1-noc_index) << " " << NOC_STATUS_READ_REG(1-noc_index, NIU_MST_WR_ACK_RECEIVED) <<ENDL();

    for (uint32_t i = 0; i < iteration; i ++) {
        uint32_t noc = noc_index;
        // uint32_t noc = (i % 2) == 0 ? noc_index : 1-noc_index;
        uint64_t noc_addr = get_noc_addr(noc_x, noc_y, l1_read_addr, noc);

        noc_async_write(l1_read_addr, noc_addr, page_size, noc);
    }
    DPRINT << "END" <<ENDL();



    while (!ncrisc_dynamic_noc_nonposted_writes_flushed<risc_type>(noc_index)) {
        uint32_t self_risc_acked = get_noc_nonposted_writes_acked<risc_type>(noc_index);
        uint32_t other_risc_acked = get_noc_nonposted_writes_acked<1-risc_type>(noc_index);

        DPRINT << "self_risc_acked " << self_risc_acked << " other_risc_acked " << other_risc_acked <<ENDL();
        DPRINT << "status reg " << NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) << " combined " << self_risc_acked + other_risc_acked <<ENDL();
    }
    noc_async_write_barrier(noc_index);
    // DPRINT << "B" <<ENDL();
    noc_async_write_barrier(1-noc_index);
    // DPRINT << "write done" <<ENDL();

}
