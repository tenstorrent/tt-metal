// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time args:
//   0: src_stride_en    - 1 = src uses 1D striding, 0 = src fixed at base
//   1: dst_stride_en    - 1 = dst uses 1D striding, 0 = dst fixed at base
//   2: num_of_addresses - total number of addresses to generate in the main loop

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

using namespace overlay;

constexpr uint32_t src_base = 0x10000;
constexpr uint64_t src_stride = 4096;  // 2 tiles × 2048 B — skips 1 tile between each access

constexpr uint32_t dst_base = 0x20000;
constexpr uint64_t dst_stride = 4096;

void kernel_main() {
    constexpr uint32_t src_stride_en = get_compile_time_arg_val(0);
    constexpr uint32_t dst_stride_en = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    setup_src_base_start_addrgen_0(src_base);
    if constexpr (src_stride_en) {
        setup_src_inner_loop_addrgen_0(src_stride, num_of_addresses * src_stride);
    }

    setup_dest_base_start_addrgen_0(dst_base);
    if constexpr (dst_stride_en) {
        setup_dest_inner_loop_addrgen_0(dst_stride, num_of_addresses * dst_stride);
    }

    /* For real NOC transfers peek/pop are not needed — replace this loop body with:
     *   push_both_addrgen_0();
     *   issue_transaction_cmdbuf_0; */
    for (uint32_t i = 0; i < num_of_addresses; ++i) {
        uint64_t src_addr = peek_src_addrgen_0();
        uint64_t dest_addr = peek_dest_addrgen_0();
        pop_src_addrgen_0();
        pop_dest_addrgen_0();
        DPRINT << "  Source address: " << HEX() << (uint32_t)src_addr << " Destination address: " << HEX()
               << (uint32_t)dest_addr << ENDL();
    }
}
