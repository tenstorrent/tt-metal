// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    constexpr uint32_t dest_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t dest_noc_y = get_compile_time_arg_val(1);
    constexpr uint32_t data_size_bytes = get_compile_time_arg_val(2);
    uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    uint32_t input_data_addr = get_arg_val<uint32_t>(0);
    // TODO: Could make the below 1 RTA
    uint32_t receiver_data_addr = get_arg_val<uint32_t>(1);
    uint32_t offset = get_arg_val<uint32_t>(2);
    static_assert(data_size_bytes <= NOC_MAX_BURST_SIZE, "Data size must be less than or equal to NOC_MAX_BURST_SIZE");
    const uint64_t dst_noc_coord = get_noc_addr(dest_noc_x, dest_noc_y, 0);
    uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(receiver_data_addr + offset);
    uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;
    noc_async_write_one_packet<true, true>(input_data_addr, dst_data_noc_addr, data_size_bytes);
    noc_semaphore_inc<true>(dst_semaphore_noc_addr, 1);
    noc_async_posted_writes_flushed();
}
