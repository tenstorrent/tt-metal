// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRISC test kernel: records the NIU mode of both NOCs into DRISC L1 so the host can check that
// firmware left NOC0 in stream mode and NOC1 in NOC2AXI mode, and that running a kernel doesn't
// disturb either.

#include "api/compile_time_args.h"
#include "experimental/drisc_mode.h"
#include "api/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t drisc_l1_dst_addr = get_compile_time_arg_val(0);

    CoreLocalMem<uint32_t> dst(drisc_l1_dst_addr);
    dst[0] = experimental::drisc_is_noc2axi_mode(0) ? 1 : 0;
    dst[1] = experimental::drisc_is_noc2axi_mode(1) ? 1 : 0;
}
