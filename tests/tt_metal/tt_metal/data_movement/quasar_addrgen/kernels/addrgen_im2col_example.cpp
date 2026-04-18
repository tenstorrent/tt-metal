// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Im2col address generation using the hardware address generator.
//
// For each output position the SW loop writes the source base address (top-left of the
// kernel window) and the HW inner/outer loops sweep the k×k kernel patch:
//
//   for (row = 0; row < out_h; ++row) {                               // SW: output rows
//     for (col = 0; col < out_w; ++col) {                             // SW: output cols
//       src_base = input + row*stride*row_bytes + col*stride*elem;
//       for (outer = 0; outer < k*dil_h*row_bytes; outer += dil_h*row_bytes) { // HW: kernel rows
//         for (inner = 0; inner < k*dil_w*elem; inner += dil_w*elem) {          // HW: kernel cols
//           yield src_base + outer + inner
//         }
//       }
//     }
//   }
//
// After k*k yields inner and outer auto-wrap to zero, so no counter reset is needed
// between output positions — just update the base.
//
// Compile-time args:
//   0: src_stride_en    - unused (striding always active for src)
//   1: dst_stride_en    - unused (striding always active for dst)
//   2: num_of_addresses - total addresses = out_h * out_w * k * k (108 for default config)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

// Image: 8 cols × 4 rows, 8 bytes per element
constexpr uint32_t img_width = 8;
constexpr uint32_t img_height = 4;
constexpr uint32_t elem = 8;
constexpr uint32_t row_bytes = img_width * elem;  // 64

// Kernel: 3×3, dilation=1 (both axes), convolution stride=1
constexpr uint32_t k = 3;
constexpr uint32_t dil_h = 1;
constexpr uint32_t dil_w = 1;
constexpr uint32_t s = 1;

// Output dimensions (valid positions only)
constexpr uint32_t out_w = img_width - (k - 1) * dil_w;   // 6
constexpr uint32_t out_h = img_height - (k - 1) * dil_h;  // 2

constexpr uint32_t src_base = 0x10000;
constexpr uint32_t dst_base = 0x200000;

// HW source loops — sweep one k×k patch per output position
//   inner: kernel columns  (fast axis)
//   outer: kernel rows     (slow axis)
constexpr uint64_t src_inner_stride = (uint64_t)dil_w * elem;        // 8
constexpr uint64_t src_inner_end = (uint64_t)k * dil_w * elem;       // 24
constexpr uint64_t src_outer_stride = (uint64_t)dil_h * row_bytes;   // 64
constexpr uint64_t src_outer_end = (uint64_t)k * dil_h * row_bytes;  // 192

// HW dest loop — linear output, one element per address
constexpr uint32_t total_addresses = out_h * out_w * k * k;  // 108
constexpr uint64_t dst_inner_stride = elem;
constexpr uint64_t dst_inner_end = (uint64_t)total_addresses * elem + 1;

void kernel_main() {
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    // Configure HW kernel-patch loops once — they auto-wrap after k*k yields
    setup_src_inner_loop_addrgen_0(src_inner_stride, src_inner_end);
    setup_src_outer_loop_addrgen_0(src_outer_stride, src_outer_end);

    // Destination: always linear
    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(dst_inner_stride, dst_inner_end);

    /* For real NOC transfers peek/pop are not needed — replace this loop body with:
     *   push_both_addrgen_0();
     *   issue_transaction_cmdbuf_0; */
    for (uint32_t row = 0; row < out_h; ++row) {
        for (uint32_t col = 0; col < out_w; ++col) {
            // Top-left of the kernel window in the input image
            uint64_t patch_base = src_base + (uint64_t)row * s * row_bytes + (uint64_t)col * s * elem;
            setup_src_base_start_addrgen_0(patch_base);

            // HW generates k*k addresses then inner/outer wrap back to zero
            for (uint32_t p = 0; p < k * k; ++p) {
                uint64_t src_addr = peek_src_addrgen_0();
                uint64_t dest_addr = peek_dest_addrgen_0();
                pop_src_addrgen_0();
                pop_dest_addrgen_0();
                DPRINT << "  [out_row=" << row << " out_col=" << col << " patch=" << p << "]"
                       << " Src: " << HEX() << (uint32_t)src_addr << " Dst: " << HEX() << (uint32_t)dest_addr << ENDL();
            }
        }
    }
}
