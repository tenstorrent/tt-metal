// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Set this DRISC's NIU mode and leave it that way.
//
// Needed because a D2H socket's config write (and its bytes_acked write-back) must reach DRISC L1
// from the host before the sender kernel ever runs. In NOC2AXI mode an inbound address in the DRAM
// range is forwarded to GDDR, and reaching L1 requires the 0x2000000000 tag -- which does not fit
// the socket's uint32_t config address. In stream mode all inbound NoC traffic terminates at L1 and
// plain local addresses work, so the socket can be constructed against a uint32_t DRISC L1 address.
//
// NIU_CFG_0 persists across programs, so whoever runs this with mode=1 owns restoring it with mode=0.

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
