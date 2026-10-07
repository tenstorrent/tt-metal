// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Bumps a per-core counter in L1 once per launch, so the host can check exactly how many times each core ran the
// program: a dropped GO leaves the count low (or hangs the wait), a duplicated GO leaves it high.
void kernel_main() {
    constexpr uint32_t counter_addr = get_compile_time_arg_val(0);
    volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr);
    counter[0] = counter[0] + 1;
}
