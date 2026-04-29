// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads from DRISC L1 into Tensix L1, handshaking via two semaphores:
//   - stream_ready (local to Tensix): Tensix waits for DRISC to remote-inc to 1.
//   - tensix_done (local to DRISC): Tensix remote-incs to 1 once the read completes.

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t tensix_dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t drisc_l1_src_addr = get_compile_time_arg_val(1);
    constexpr uint32_t drisc_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t drisc_noc_y = get_compile_time_arg_val(3);
    constexpr uint32_t stream_ready_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t tensix_done_sem_id = get_compile_time_arg_val(5);

    experimental::Noc noc;

    // Wait locally for DRISC to signal stream mode is ready.
    experimental::Semaphore<ProgrammableCoreType::TENSIX> stream_ready(stream_ready_sem_id);
    stream_ready.wait(1);

    // Read one uint32_t from DRISC L1 into Tensix L1.
    experimental::UnicastEndpoint src;
    experimental::CoreLocalMem<uint32_t> dst(tensix_dst_addr);
    noc.async_read(
        src, dst, sizeof(uint32_t), {.noc_x = drisc_noc_x, .noc_y = drisc_noc_y, .addr = drisc_l1_src_addr}, {});
    noc.async_read_barrier();

    // Signal DRISC (remote inc to 1) that the read is complete, and flush the
    // non-posted atomic before kernel exit so watcher doesn't flag a pending txn.
    experimental::Semaphore<ProgrammableCoreType::DRAM> tensix_done(tensix_done_sem_id);
    tensix_done.up(noc, drisc_noc_x, drisc_noc_y, 1);
    noc.async_atomic_barrier();
}
