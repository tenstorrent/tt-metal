// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time args:
//   0: src_stride_en    - 1 = src uses 2D striding, 0 = src fixed at base
//   1: dst_stride_en    - 1 = dst uses 2D striding, 0 = dst fixed at base
//   2: num_of_addresses - total number of addresses to generate in the main loop
//                         must equal inner_count * outer_count (4 * 4 = 16 for the default config)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

// 4 cols x 4 rows
constexpr uint32_t src_base = 0x30000;
constexpr LoopConfig src_inner_cfg = {.stride = 128, .end_addr = 4 * 128};    // 4 cols, 128B apart
constexpr LoopConfig src_outer_cfg = {.stride = 1024, .end_addr = 4 * 1024};  // 4 rows, 1024B apart

constexpr uint32_t dst_base = 0x40000;
constexpr LoopConfig dst_inner_cfg = {.stride = 128, .end_addr = 4 * 128};
constexpr LoopConfig dst_outer_cfg = {.stride = 1024, .end_addr = 4 * 1024};

void kernel_main() {
    constexpr uint32_t src_stride_en = get_compile_time_arg_val(0);
    constexpr uint32_t dst_stride_en = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    setup_src_base_start_addrgen_0(src_base);
    if constexpr (src_stride_en) {
        setup_src_inner_loop_addrgen_0(src_inner_cfg);
        setup_src_outer_loop_addrgen_0(src_outer_cfg);
    }

    setup_dest_base_start_addrgen_0(dst_base);
    if constexpr (dst_stride_en) {
        setup_dest_inner_loop_addrgen_0(dst_inner_cfg);
        setup_dest_outer_loop_addrgen_0(dst_outer_cfg);
    }

    for (uint32_t i = 0; i < num_of_addresses; ++i) {
        /* If used with NOC, there is no need to first read addresses from address generator */
        uint64_t src_addr = peek_src_addrgen_0();
        uint64_t dest_addr = peek_dest_addrgen_0();
        /* We need to use push if we need addresses in cmd buffers
        if not needed, pop will just make hw generate new address and discard them */
        // push_both_addrgen_0();
        /* This will issue the transaction command buffer with the addresses */
        // issue_transaction_cmdbuf_0
        pop_src_addrgen_0();
        pop_dest_addrgen_0();
        DPRINT << "  Source address: " << HEX() << (uint32_t)src_addr << " Destination address: " << HEX()
               << (uint32_t)dest_addr << ENDL();
    }
}
// /* Real loop with noc */
// /* configure noc(vc,trid,read,write..) */
// for (uint32_t i = 0; i < num_of_addresses; ++i) {
//     push_both_addrgen_0();
//     /* there are los push_src and push_dst if only one is needed */
//     issue_transaction_cmdbuf_0
// }
