// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC test kernel. Two test paths:
//   with MODE_TENSIX_TO_DRISC defined (Tensix-to-DRISC read path):
//     DRISC enters stream mode, reads a uint32_t from Tensix L1 into DRISC L1,
//     then restores NOC2AXI.
//   default (DRISC-to-Tensix handshake path):
//     DRISC writes magic_value into its L1 after switching to stream mode, remote-bumps Tensix's stream_ready
//     (-> 1) to signal readiness, waits locally on tensix_done until Tensix
//     finishes reading, then restores NOC2AXI.

#include "api/compile_time_args.h"
#include "experimental/drisc_mode.h"
#include "experimental/noc.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t drisc_l1_dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t tensix_noc_x = get_compile_time_arg_val(1);
    constexpr uint32_t tensix_noc_y = get_compile_time_arg_val(2);

    // Stream mode: required for DRISC to initiate NOC traffic and for
    // remote cores to reach DRISC L1 over NOC.
    experimental::drisc_set_stream_mode();

    experimental::Noc noc;

#ifdef MODE_TENSIX_TO_DRISC
    uint32_t tensix_l1_src_addr = get_arg_val<uint32_t>(0);

    experimental::UnicastEndpoint src;
    experimental::CoreLocalMem<uint32_t> dst(drisc_l1_dst_addr);
    noc.async_read(
        src, dst, sizeof(uint32_t), {.noc_x = tensix_noc_x, .noc_y = tensix_noc_y, .addr = tensix_l1_src_addr}, {});
    noc.async_read_barrier();
#else
    uint32_t stream_ready_sem_id = get_arg_val<uint32_t>(0);
    uint32_t tensix_done_sem_id = get_arg_val<uint32_t>(1);
    uint32_t magic_value = get_arg_val<uint32_t>(2);

    // Write magic value into DRISC L1 for Tensix to read back.
    experimental::CoreLocalMem<uint32_t> dst(drisc_l1_dst_addr);
    dst[0] = magic_value;

    // Signal Tensix (remote inc to 1) that stream mode is on and DRISC L1 is ready.
    experimental::Semaphore<ProgrammableCoreType::TENSIX> stream_ready(stream_ready_sem_id);
    stream_ready.up(noc, tensix_noc_x, tensix_noc_y, 1);
    noc.async_atomic_barrier();

    // Wait locally for Tensix to confirm completion.
    experimental::Semaphore<ProgrammableCoreType::DRAM> tensix_done(tensix_done_sem_id);
    tensix_done.wait(1);
#endif

    // Always restore NOC2AXI so subsequent context observes the default.
    experimental::drisc_set_noc2axi_mode();
}
