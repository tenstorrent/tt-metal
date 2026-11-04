// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK assembly: the mhc's ONCE-PER-PROGRAM init state for a program that runs a LATER mhc part (A5 = P10 without P1):
// the kernels' init() (F_INIT, part P1 only) reserves and zeroes CBs that are "reserved once and never pushed" -- the
// members' PRE (4 tiles) / INV (1), the root's PLANE (1) / SCR (2) / PCOEF (NRB x 8). Same RISC as the mhc sync part
// (NCRISC), before it. Runtime args at ph_rt0: [n, (cb, pages, bytes) x n].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    (void)PH_CT0;
    const uint32_t a = ph_rt0;
    const uint32_t n = get_arg_val<uint32_t>(a + 0);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t cb = get_arg_val<uint32_t>(a + 1 + 3 * i);
        const uint32_t pages = get_arg_val<uint32_t>(a + 2 + 3 * i);
        const uint32_t bytes = get_arg_val<uint32_t>(a + 3 + 3 * i);
        cb_reserve_back(cb, pages);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
        for (uint32_t w = 0; w < bytes / 4; w++) {
            p[w] = 0;
        }
    }
}
