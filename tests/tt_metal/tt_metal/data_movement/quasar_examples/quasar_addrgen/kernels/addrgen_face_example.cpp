// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Demonstrates the face loop: the outermost loop in the address generator hierarchy.
//
// Hardware loop structure:
//   for (base = base_start; ; base += face_size) {          // face loop (infinite, no end)
//     for (outer = 0; outer < outer_end; outer += stride) { // outer loop
//       for (inner = 0; inner < inner_end; inner += stride) { // inner loop
//         yield base + outer + inner
//       }
//     }
//   }
//
// After each complete inner x outer iteration (one "face"), the base address advances
// by face_size. This is useful for processing multiple tiles laid out contiguously.
//
// Compile-time args:
//   0: src_stride_en    - 1 = src uses face+2D striding, 0 = src fixed at base
//   1: dst_stride_en    - 1 = dst uses face+2D striding, 0 = dst fixed at base
//   2: num_of_addresses - total addresses to generate; should be num_faces * outer_count * inner_count

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

// 2 faces, each face: 4 cols x 4 rows
constexpr uint32_t src_base = 0x10000;
constexpr LoopConfig src_inner_cfg = {.stride = 128, .end_addr = 4 * 128};    // 4 cols, 128B apart
constexpr LoopConfig src_outer_cfg = {.stride = 1024, .end_addr = 4 * 1024};  // 4 rows, 1024B apart
constexpr uint64_t src_face_size = src_outer_cfg.end_addr;                    // one face = 4096B

constexpr uint32_t dst_base = 0x20000;
constexpr LoopConfig dst_inner_cfg = {.stride = 128, .end_addr = 4 * 128};
constexpr LoopConfig dst_outer_cfg = {.stride = 1024, .end_addr = 4 * 1024};
constexpr uint64_t dst_face_size = dst_outer_cfg.end_addr;

void kernel_main() {
    constexpr uint32_t src_stride_en = get_compile_time_arg_val(0);
    constexpr uint32_t dst_stride_en = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    setup_src_base_start_addrgen_0(src_base);
    if constexpr (src_stride_en) {
        /* After outer loop, base address is advanced by face size */
        setup_src_face_size_addrgen_0(src_face_size);
        setup_src_inner_loop_addrgen_0(src_inner_cfg);
        setup_src_outer_loop_addrgen_0(src_outer_cfg);
    }

    setup_dest_base_start_addrgen_0(dst_base);
    if constexpr (dst_stride_en) {
        /* After outer loop, base address is advanced by face size */
        setup_dest_face_size_addrgen_0(dst_face_size);
        setup_dest_inner_loop_addrgen_0(dst_inner_cfg);
        setup_dest_outer_loop_addrgen_0(dst_outer_cfg);
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
// /* Real loop with noc */
// /* configure noc(vc,trid,read,write..) */
// for (uint32_t i = 0; i < num_of_addresses; ++i) {
//     push_both_addrgen_0();
//     /* there are los push_src and push_dst if only one is needed */
//     issue_transaction_cmdbuf_0
// }
