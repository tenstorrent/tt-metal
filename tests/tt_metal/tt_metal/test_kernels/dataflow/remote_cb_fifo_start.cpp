// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reports where firmware set this core's remote circular buffer up: its fifo_start_addr, which is 0 when the
// circular buffer is absent on this core.
//
// Args (compile time): [0] remote circular buffer index
// Args (runtime):      [0] L1 address the word is written to

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    constexpr uint32_t remote_cb_index = get_compile_time_arg_val(0);
    const uint32_t result_addr = get_arg_val<uint32_t>(0);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr) =
        get_remote_receiver_cb_interface(remote_cb_index).fifo_start_addr;
}
