// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t src_addr           = get_arg_val<uint32_t>(0);
    uint32_t bank_id            = get_arg_val<uint32_t>(1);
    uint32_t src_buffer_size    = get_arg_val<uint32_t>(2);

    uint32_t local_addr         = get_arg_val<uint32_t>(3);

    uint32_t dst_addr           = get_arg_val<uint32_t>(4);
    uint32_t dst_noc_x_start    = get_arg_val<uint32_t>(5);
    uint32_t dst_noc_y_start    = get_arg_val<uint32_t>(6);
    uint32_t dst_noc_x_end      = get_arg_val<uint32_t>(7);
    uint32_t dst_noc_y_end      = get_arg_val<uint32_t>(8);
    uint32_t num_dests          = get_arg_val<uint32_t>(9);

    // Read src buffer into local L1 buffer
    constexpr auto bank_type = experimental::AllocatorBankType::DRAM;
    experimental::CoreLocalMem<std::uint32_t> local_buffer(local_addr);

    experimental::Noc noc;
    noc.async_read(
        experimental::AllocatorBank<bank_type>(),
        local_buffer,
        src_buffer_size,
        {.bank_id = bank_id, .addr = src_addr},
        {});
    noc.async_read_barrier();

    // multicast local L1 buffer to all destination cores
    experimental::MulticastEndpoint dst_mcast_endpoint;
    noc.async_write_multicast<experimental::Noc::McastMode::INCLUDE_SRC>(
        local_buffer,
        dst_mcast_endpoint,
        src_buffer_size,
        num_dests,
        {},
        {.noc_x_start = dst_noc_x_start,
         .noc_y_start = dst_noc_y_start,
         .noc_x_end = dst_noc_x_end,
         .noc_y_end = dst_noc_y_end,
         .addr = dst_addr});
    noc.async_write_barrier();
}
