// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone D2D "sync" op — the gate a real model graph inserts BEFORE the op
// that overwrites the outbound D2DStreamServiceSender backing tensor.
//
// In production the producing op (a matmul / CCL) OWNS the data_ready signal (it
// incs the counter itself, after it runs) but cannot self-gate: it must not
// overwrite the outbound tensor until the sender service has forwarded the
// previous iteration. Since a black-box op can't spin a semaphore before it runs,
// that gate is a SEPARATE op — this kernel. It waits on the sender service's
// consumed_sem (the service multicast-incs it after forwarding) and resets it.
//
// Runs on the SAME worker grid as the producing op so it consumes exactly the
// num_workers multicast increments (one per worker core). One gate per launch;
// the host relaunches it each iter (except iter 0, which has no prior forward).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t consumed_sem_addr = get_compile_time_arg_val(0);

void kernel_main() {
    volatile tt_l1_ptr uint32_t* consumed_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_sem_addr);
    while (*consumed_sem == 0) {
        invalidate_l1_cache();
    }
    *consumed_sem = 0;
}
