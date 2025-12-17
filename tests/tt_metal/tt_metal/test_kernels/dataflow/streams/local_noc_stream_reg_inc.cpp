// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "noc_overlay_parameters.h"

void kernel_main() {
#if defined(COMPILE_FOR_ERISC) or defined(COMPILE_FOR_IDLE_ERISC)
    constexpr uint32_t num_streams = ETH_NOC_NUM_STREAMS;
#else
    constexpr uint32_t num_streams = NOC_NUM_STREAMS;
#endif
    uint32_t read_value = 0;
    volatile tt_l1_ptr uint32_t* status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    for (uint32_t i = 0; i < num_streams; i++) {
        uint32_t initial_value = i;
        uint32_t increment_value = i;
        uint32_t expected_value = initial_value + increment_value;

        NOC_STREAM_WRITE_REG(i, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, initial_value);
        read_value =
            NOC_STREAM_READ_REG(i, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX) & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
        if (read_value != initial_value) {
            *status_ptr = 1;
            return;
        }

        read_value = NOC_STREAM_READ_REG(i, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
                     ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
        if (read_value != initial_value) {
            *status_ptr = 2;
            return;
        }

        NOC_STREAM_WRITE_REG(
            i,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            increment_value << REMOTE_DEST_BUF_WORDS_FREE_INC);
        for (uint32_t retry = 0; retry < 1000; retry++) {
            read_value = NOC_STREAM_READ_REG(i, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
                         ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
            if (read_value == expected_value) {
                break;
            }
        }
        if (read_value != expected_value) {
            *status_ptr = 3;
            return;
        }
    }
    *status_ptr = 0;
}
