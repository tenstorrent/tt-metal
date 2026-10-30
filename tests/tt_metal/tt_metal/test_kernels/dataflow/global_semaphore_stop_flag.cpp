// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore as a stop flag; the semaphore's address is common runtime arg 0. Runtime args {report_addr}.
// Spin on value() until the host writes a non-zero value (bounded), then report {value seen, 1 if the spin hit
// its cap}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"

constexpr GlobalSemaphoreBindingToken stop_token{0};

void kernel_main() {
    const uint32_t report_addr = get_arg_val<uint32_t>(0);

    GlobalSemaphore stop(stop_token);
    constexpr uint32_t kSpinCap = 1u << 28;
    uint32_t spins = 0;
    while (stop.value() == 0 && spins < kSpinCap) {
        spins++;
    }

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = stop.value();
    report[1] = spins == kSpinCap ? 1u : 0u;
}
