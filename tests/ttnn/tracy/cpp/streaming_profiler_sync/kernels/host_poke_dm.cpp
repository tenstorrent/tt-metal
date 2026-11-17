// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Each round, the host writes the round number to a flag in this core's L1. The kernel records a HOST_RX zone when the
// new round appears, then acks it.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "sync_workload.hpp"

void kernel_main() {
    using namespace sync_workload;
    const uint32_t flag_addr = get_arg_val<uint32_t>(0);
    const uint32_t rounds = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    for (uint32_t round = 1; round <= rounds; round++) {
        if (!wait_for_round(flag, round)) {
            return;
        }
        {
            DeviceZoneScopedN("HOST_RX");
        }
        flag[kAckWord] = round;
    }
}
