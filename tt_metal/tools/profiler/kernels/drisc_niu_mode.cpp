// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Set this DRISC's NIU mode and leave it. The relay's socket config must reach DRISC L1 from the host before
// the kernel runs; in NOC2AXI mode a DRAM-range address forwards to GDDR and reaching L1 needs the
// 0x2000000000 tag, which does not fit the socket's uint32_t address, while in stream mode inbound traffic
// terminates at L1. NIU_CFG_0 persists across programs, so whoever sets mode=1 restores mode=0.

#include "api/compile_time_args.h"
#include "experimental/drisc_mode.h"

void kernel_main() {
    constexpr uint32_t kStreamMode = get_compile_time_arg_val(0);
    if constexpr (kStreamMode) {
        experimental::drisc_set_stream_mode_all();
    } else {
        experimental::drisc_set_noc2axi_mode_all();
    }
}
