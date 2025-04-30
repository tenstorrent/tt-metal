// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t target_noc_x = get_arg_val<uint32_t>(0);
    uint32_t target_noc_y = get_arg_val<uint32_t>(1);
    uint32_t stream_id = get_arg_val<uint32_t>(2);
    uint32_t target_core_value = get_arg_val<uint32_t>(3);
    uint32_t semaphore_addr = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t multicast_start_x = get_arg_val<uint32_t>(5);
    uint32_t multicast_end_x = get_arg_val<uint32_t>(6);
    uint32_t multicast_start_y = get_arg_val<uint32_t>(7);
    uint32_t multicast_end_y = get_arg_val<uint32_t>(8);
    uint32_t num_dests = get_arg_val<uint32_t>(9);

    volatile tt_l1_ptr uint32_t* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    if (target_core_value) {
        // Clear the stream register and signal the other cores it's safe to send.
        NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
        *semaphore_ptr = 1;
        uint64_t multicast_data_addr = get_noc_multicast_addr(
            multicast_start_x, multicast_start_y, multicast_end_x, multicast_end_y, semaphore_addr);
        noc_semaphore_set_multicast(semaphore_addr, multicast_data_addr, num_dests);
    }

    noc_semaphore_wait(semaphore_ptr, 1);

    // Write to stream register at `reg_addr` on core [target_noc_x, target_noc_y]
    uint32_t reg_addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    uint64_t dest_addr = NOC_XY_ADDR(target_noc_x, target_noc_y, reg_addr);
    noc_inline_dw_write<true>(dest_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);

    if (target_core_value) {
        while (target_core_value != (NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
                                     ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1))) {
        }
    }

    noc_async_writes_flushed();
    if (target_core_value) {
        noc_async_write_barrier();
    }
}
