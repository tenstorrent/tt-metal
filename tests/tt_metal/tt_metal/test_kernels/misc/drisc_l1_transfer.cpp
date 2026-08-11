// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC test kernel: enters stream mode, reads a uint32_t from Tensix L1 into DRISC L1, restores NOC2AXI.

#include "api/compile_time_args.h"
#include "experimental/drisc_mode.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t drisc_l1_dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t tensix_noc_x = get_compile_time_arg_val(1);
    constexpr uint32_t tensix_noc_y = get_compile_time_arg_val(2);

    Noc noc;

    // Stream mode: required for DRISC to initiate NOC traffic and for
    // remote cores to reach DRISC L1 over NOC.
    experimental::drisc_set_stream_mode();
    uint32_t tensix_l1_src_addr = get_arg_val<uint32_t>(0);

    UnicastEndpoint src;
    CoreLocalMem<uint32_t> dst(drisc_l1_dst_addr);
    noc.async_read(
        src, dst, sizeof(uint32_t), {.noc_x = tensix_noc_x, .noc_y = tensix_noc_y, .addr = tensix_l1_src_addr}, {});
    noc.async_read_barrier();

    // Always restore NOC2AXI so subsequent context observes the default.
    experimental::drisc_set_noc2axi_mode();
}
