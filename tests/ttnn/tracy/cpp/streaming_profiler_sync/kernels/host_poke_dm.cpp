// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The host sync test's kernel: for each round the host writes the round number to the flag, and the kernel records an
// HOST_RX zone when it sees it, then acks the round.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "workload_layout.hpp"

void kernel_main() {
    const uint32_t flag_addr = get_arg_val<uint32_t>(0);
    const uint32_t rounds = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    for (uint32_t round = 1; round <= rounds; round++) {
        for (uint32_t polls = 0; flag[kRoundWord] != round; polls++) {
            invalidate_l1_cache();
            if (polls == kSpinLimit) {
                flag[kGaveUpRoundWord] = round;
                return;
            }
        }
        {
            DeviceZoneScopedN("HOST_RX");
        }
        flag[kAckWord] = round;
    }
}
