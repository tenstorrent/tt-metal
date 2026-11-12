// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Records the NoC ID registers a core can read about itself, so the host can check which coordinate system each
// holds. Runs on Tensix, ERISC and DRISC.
// arg 0: local L1 address for 4 words: NOC_NODE_ID on NOC0, NOC_ID_LOGICAL on NOC0, then the same two on NOC1
// arg 1: nonzero also reads the NOC1 registers (otherwise their words are 0xFFFFFFFF)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t read_noc1 = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_addr);
    for (uint32_t noc = 0; noc < 2; noc++) {
        const bool read = noc == 0 || read_noc1 != 0;
        out[2 * noc] = read ? NOC_CMD_BUF_READ_REG(noc, 0, NOC_NODE_ID) : 0xFFFFFFFF;
        out[2 * noc + 1] = read ? NOC_CMD_BUF_READ_REG(noc, 0, NOC_CFG(NOC_ID_LOGICAL)) : 0xFFFFFFFF;
    }
}
