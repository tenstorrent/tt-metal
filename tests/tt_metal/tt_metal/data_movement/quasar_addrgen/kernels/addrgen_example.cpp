// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time args:
//   0: src_mode        - 0 = 1D strided, 1 = 2D strided
//   1: dst_mode        - 0 = 1D strided, 1 = 2D strided
//   2: num_of_addresses - total number of addresses to generate in the main loop

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

// 1D source config
constexpr uint32_t src_1d_base = 0x10000;
constexpr uint64_t src_1d_stride = 2048;

// 2D source config: 4 rows x 4 cols
constexpr uint32_t src_2d_base = 0x30000;
constexpr LoopConfig src_2d_inner_cfg = {.stride = 128, .end_addr = 4 * 128};    // 4 cols, 128B apart
constexpr LoopConfig src_2d_outer_cfg = {.stride = 1024, .end_addr = 4 * 1024};  // 4 rows, 1024B apart

// 1D destination config
constexpr uint32_t dst_1d_base = 0x20000;
constexpr uint64_t dst_1d_stride = 2048;

// 2D destination config: 4 rows x 4 cols
constexpr uint32_t dst_2d_base = 0x40000;
constexpr LoopConfig dst_2d_inner_cfg = {.stride = 128, .end_addr = 4 * 128};
constexpr LoopConfig dst_2d_outer_cfg = {.stride = 1024, .end_addr = 4 * 1024};

void kernel_main() {
    constexpr uint32_t src_mode = get_compile_time_arg_val(0);          // 0 = 1D strided, 1 = 2D strided
    constexpr uint32_t dst_mode = get_compile_time_arg_val(1);          // 0 = 1D strided, 1 = 2D strided
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);  // total addresses to generate

    reset_addrgen_0();

    // --- Source setup ---
    // 1D strided source
    if constexpr (src_mode == 0) {
        constexpr LoopConfig src_1d_cfg = {.stride = src_1d_stride, .end_addr = num_of_addresses * src_1d_stride};
        setup_src_base_start_addrgen_0(src_1d_base);
        setup_src_inner_loop_addrgen_0(src_1d_cfg.stride, src_1d_cfg.end_addr);
    }
    // 2D strided source
    else if constexpr (src_mode == 1) {
        setup_src_base_start_addrgen_0(src_2d_base);
        setup_src_inner_loop_addrgen_0(src_2d_inner_cfg.stride, src_2d_inner_cfg.end_addr);
        setup_src_outer_loop_addrgen_0(src_2d_outer_cfg.stride, src_2d_outer_cfg.end_addr);
    }

    // --- Destination setup ---
    // 1D strided destination
    if constexpr (dst_mode == 0) {
        constexpr LoopConfig dst_1d_cfg = {.stride = dst_1d_stride, .end_addr = num_of_addresses * dst_1d_stride};
        setup_dest_base_start_addrgen_0(dst_1d_base);
        setup_dest_inner_loop_addrgen_0(dst_1d_cfg.stride, dst_1d_cfg.end_addr);

    }
    // 2D strided destination
    else if constexpr (dst_mode == 1) {
        setup_dest_base_start_addrgen_0(dst_2d_base);
        setup_dest_inner_loop_addrgen_0(dst_2d_inner_cfg);
        setup_dest_outer_loop_addrgen_0(dst_2d_outer_cfg);
    }

    /* Main loop to generate addresses */
    for (uint32_t i = 0; i < num_of_addresses; ++i) {
        uint64_t src_addr = peek_src_addrgen_0();
        uint64_t dest_addr = peek_dest_addrgen_0();
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
