// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sender kernel for NOC hop latency ping-pong benchmark.
// Sends a semaphore inc to responder, waits for responder to inc our semaphore back.
// Records per-iteration wall clock cycle deltas in an L1 timestamp buffer.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"

// risc_common.h has get_timestamp() but we just use reg_read directly for both ERISC and tensix

void kernel_main() {
    set_l1_data_cache<false>();
    uint32_t arg_idx = 0;
    const uint32_t responder_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t responder_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t responder_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t timestamp_buf_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_iterations = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_warmup = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ready_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_sem_addr);
    volatile tt_l1_ptr uint32_t* ts_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timestamp_buf_addr);

    *local_sem = 0;

    // If ready_sem_addr is nonzero, wait for responder to signal it's ready
    if (ready_sem_addr != 0) {
        volatile tt_l1_ptr uint32_t* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem_addr);
        *ready_sem = 0;
        while (*ready_sem == 0) {
        }
    }

    uint64_t responder_sem_noc_addr = get_noc_addr(responder_noc_x, responder_noc_y, responder_sem_addr);

    // Warmup iterations (not timed)
    for (uint32_t i = 0; i < num_warmup; i++) {
        noc_semaphore_inc(responder_sem_noc_addr, 1);
        while (*local_sem == 0) {
        }
        *local_sem = 0;
    }

    // Timed iterations — store [issue_time, poll_time] pairs
    for (uint32_t i = 0; i < num_iterations; i++) {
        uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);

        noc_semaphore_inc(responder_sem_noc_addr, 1);
        uint32_t seminc_issue_end = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);

        noc_semaphore_wait(local_sem, 1);
        uint32_t t1 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        noc_semaphore_set(local_sem, 0);

        ts_buf[2 * i] = seminc_issue_end - t0;      // issue time (noc_semaphore_inc call)
        ts_buf[2 * i + 1] = t1 - seminc_issue_end;  // poll time (pure wait for response)
    }
}
