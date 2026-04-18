// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Im2Col with dilation support, direct port of im2col.cpp to the addrgen API.
//
// The HW inner/outer loops cover one kernel patch (dilation-aware strides).
// SW vertical and horizontal loops slide the kernel window over all output positions,
// writing src_base before each patch — matching CMDBUF_WR_REG(...SRC_BASE...) in the original.
//
// HW loop structure per output position:
//   for (outer = 0; outer < eff_kH * row_bytes; outer += row_bytes * dil_h) {  // kernel rows
//     for (inner = 0; inner < k * dil_w * elem; inner += elem * dil_w) {        // kernel cols
//       yield src_base + outer + inner
//     }
//   }
//
// Image:    H x W = 6 x 7,  elem_size = 8 B
// Kernel:   k x k = 3 x 3,  dilation_h = 1,  dilation_w = 2,  stride = 1
// eff_kW  = k + (k-1)*(dil_w-1) = 5
// eff_kH  = k + (k-1)*(dil_h-1) = 3
// SW iters: (H - eff_kH) vertical  x  (W - eff_kW + 1) horizontal  =  3 x 3
// Total addresses: 3 * 3 * k*k = 81
//
// Compile-time args:
//   0: (unused)
//   1: (unused)
//   2: num_of_addresses — must equal vertical_iters * horz_iters * k * k

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

constexpr uint32_t H = 6;
constexpr uint32_t W = 7;
constexpr uint32_t k = 3;
constexpr uint32_t dil_h = 1;
constexpr uint32_t dil_w = 2;
constexpr uint32_t stride = 1;
constexpr uint32_t elem_size = 8;
constexpr uint32_t row_bytes = W * elem_size;  // 56

constexpr uint32_t eff_kW = k + (k - 1) * (dil_w - 1);  // 5
constexpr uint32_t eff_kH = k + (k - 1) * (dil_h - 1);  // 3

constexpr uint32_t src_base = 0x10000;
constexpr uint32_t dst_base = 0x20000;

// HW source loops (configured once, auto-wrap after k*k yields)
constexpr uint64_t inner_stride = (uint64_t)elem_size * dil_w;           // 16
constexpr uint64_t inner_end = (uint64_t)k * dil_w * elem_size;          // 48
constexpr uint64_t outer_stride = (uint64_t)row_bytes * stride * dil_h;  // 56
constexpr uint64_t outer_end = (uint64_t)eff_kH * stride * row_bytes;    // 168

// SW loop bounds (byte offsets, matching original formulas exactly)
constexpr uint64_t vert_stride = (uint64_t)row_bytes * stride * dil_h;         // 56
constexpr uint64_t vert_end = (uint64_t)(H - eff_kH) * vert_stride;            // 168
constexpr uint64_t horz_stride = (uint64_t)stride * elem_size;                 // 8
constexpr uint64_t horz_end = (uint64_t)(W - eff_kW) * elem_size + elem_size;  // 24

constexpr uint32_t total_addresses =
    ((uint32_t)(vert_end / vert_stride)) * ((uint32_t)(horz_end / horz_stride)) * k * k;  // 3 * 3 * 9 = 81

void kernel_main() {
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    // HW loops for one kernel patch — configured once, wrap naturally after k*k pops
    setup_src_inner_loop_addrgen_0(inner_stride, inner_end);
    setup_src_outer_loop_addrgen_0(outer_stride, outer_end);

    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(elem_size, (uint64_t)total_addresses * elem_size);

    /* For real NOC transfers peek/pop are not needed — replace this loop body with:
     *   push_both_addrgen_0();
     *   issue_transaction_cmdbuf_0; */
    for (uint64_t v = 0; v < vert_end; v += vert_stride) {
        for (uint64_t h = 0; h < horz_end; h += horz_stride) {
            setup_src_base_start_addrgen_0(src_base + v + h);

            for (uint32_t p = 0; p < k * k; ++p) {
                uint64_t src_addr = peek_src_addrgen_0();
                uint64_t dest_addr = peek_dest_addrgen_0();
                pop_src_addrgen_0();
                pop_dest_addrgen_0();
                DPRINT << "  Source address: " << HEX() << (uint64_t)src_addr << " Destination address: " << HEX()
                       << (uint64_t)dest_addr << ENDL();
            }
        }
    }
}
