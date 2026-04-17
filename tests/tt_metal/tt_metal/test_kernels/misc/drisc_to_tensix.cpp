// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC kernel that puts a local NIU into stream mode and then
// reads from the Tensix's L1.

#include "api/compile_time_args.h"
#include "experimental/drisc_mode.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t tensix_l1_src_addr = get_compile_time_arg_val(0);
    constexpr uint32_t drisc_l1_dst_addr = get_compile_time_arg_val(1);
    constexpr uint32_t tensix_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t tensix_noc_y = get_compile_time_arg_val(3);

    // To enable DRISC to initiate NOC transactions to Tensix L1,
    // we need to configure DRISC's NIU into stream mode from default NOC2AXI mode
    drisc_set_stream_mode();
    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> dst(drisc_l1_dst_addr);
    experimental::UnicastEndpoint src;
    noc.async_read(
        src, dst, sizeof(uint32_t), {.noc_x = tensix_noc_x, .noc_y = tensix_noc_y, .addr = tensix_l1_src_addr}, {});

    noc.async_read_barrier();
}
