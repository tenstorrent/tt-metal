// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "internal/tt-1xx/blackhole/gddr_mc_regs.h"

void kernel_main() {
    const uint32_t result_l1_addr = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr);

    for (uint32_t port = 1; port <= 3; ++port) {
        gddr_mc_write_mpfe_weight(port, get_arg_val<uint32_t>(port - 1));
        results[port - 1] = gddr_mc_read_mpfe_weight(port);
    }
}
