// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix kernel that puts a remote DRISC NIU into stream mode and then
// reads from the DRISC's L1.

#include "api/compile_time_args.h"
#include "drisc_mode.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t tensix_dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t drisc_l1_src_addr = get_compile_time_arg_val(1);
    constexpr uint32_t drisc_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t drisc_noc_y = get_compile_time_arg_val(3);

    // Reuse tensix_dst_addr as scratch for NIU_CFG_0 readback. Safe because
    // every remote API call finishes (via barrier) before the next one
    // clobbers the slot, and the final noc_async_read overwrites it with
    // the DRISC L1 value we actually care about.
    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> dst(tensix_dst_addr);
    experimental::UnicastEndpoint src;
    noc.async_read(
        src, dst, sizeof(uint32_t), {.noc_x = drisc_noc_x, .noc_y = drisc_noc_y, .addr = drisc_l1_src_addr}, {});

    noc.async_read_barrier();
}
