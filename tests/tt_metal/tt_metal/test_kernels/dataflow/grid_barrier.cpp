// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Grid all-to-one barrier. Every non-target node does one remote hardware-atomic increment of the
// target node's barrier semaphore over the NoC; the target drains all num_signalers increments with
// a single down(num_signalers) (race-free wait-for-threshold) then wait(0) (confirms EXACTLY N
// arrived), and only then records a RELEASED sentinel in L1 for the host to verify. Exercises max
// semaphore fan-in onto a single counter. Non-DFB: pure cross-node semaphore + NoC.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t target_noc_x = get_arg(args::remote_noc_x);
    const uint32_t target_noc_y = get_arg(args::remote_noc_y);
    const uint32_t is_target = get_arg(args::is_target);
    const uint32_t num_signalers = get_arg(args::num_elements);
    const uint32_t result_addr = get_arg(args::result_addr);

    Noc noc;
    Semaphore barrier_sem(sem::barrier_sem);

    if (is_target) {
        // Wait for all signalers with a SINGLE decrement: down(N) blocks until the counter
        // reaches N, then subtracts N once. This avoids the lost-update race of N separate
        // non-atomic down(1) read-modify-writes racing the signalers' hardware-atomic
        // increments; and because there are exactly num_signalers signalers, no increment
        // arrives after the threshold, so the single subtract is race-free.
        barrier_sem.down(num_signalers);
        // Confirm EXACTLY num_signalers arrived: after subtracting N the counter must be 0. A
        // stray/duplicate increment (over-count) leaves it > 0 and wait(0) hangs (detected).
        barrier_sem.wait(0);
        // Record RELEASED only once both checks pass -- an OBSERVED property, not an input echo.
        CoreLocalMem<uint32_t> result(result_addr);
        result[0] = 0xC0DEBA11u;
        flush_l2_cache_line(result_addr);
    } else {
        // Signal the target exactly once (hardware-atomic remote increment).
        barrier_sem.up(noc, target_noc_x, target_noc_y, 1);
    }
}
