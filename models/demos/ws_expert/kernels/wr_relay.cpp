// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert store-and-forward: group slot credit fan-out (BRISC, on-chip only). This chip's group cores write their
// freed-expert counts into this core's words c (c = g * GC + j); every change goes to word c of each lander on this
// chip (the ring forwarders and the local lander), which wait on it before writing a slice into the
// core's ring slot. Slot credits never leave the chip: the ring credits are per hop (wr_fwd.cpp).
// RT: 0.. NLAND lander xy (x | y << 8)
// CT: 0 NC, 1 GC, 2 G, 3 NEXP, 4 icred, 5 NLAND
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0), GC = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2), NEXP = get_compile_time_arg_val(3);
    constexpr uint32_t icred = get_compile_time_arg_val(4), NLAND = get_compile_time_arg_val(5);
    uint64_t dst[NLAND];
    for (uint32_t i = 0; i < NLAND; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(i);
        dst[i] = get_noc_addr(xy & 0xFF, xy >> 8, icred);
    }
    uint32_t fwd[NC];
    for (uint32_t c = 0; c < NC; ++c) {
        fwd[c] = 0;
    }
    uint32_t done = 0;
    while (done < NC) {
        invalidate_l1_cache();
        for (uint32_t c = 0; c < NC; ++c) {
            const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + c * 16);
            if (v <= fwd[c]) {
                continue;
            }
            for (uint32_t i = 0; i < NLAND; ++i) {
                noc_inline_dw_write(dst[i] + c * 16, v);
            }
            fwd[c] = v;
            const uint32_t g = c / GC;
            if (v >= (NEXP - g + G - 1) / G) {  // this core's last expert
                ++done;
            }
        }
    }
    noc_async_full_barrier();
}
