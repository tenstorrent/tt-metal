// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Im2col optimised for dilation=1 using the hardware face loop.
//
// When dilation=1 the HW face loop can replace the SW output-row loop: after each
// complete inner×outer sweep (one k×k patch) the hardware automatically advances the
// source base address by face_stride (= row_bytes), moving to the next output row.
// Only the output-column SW loop remains, calling reset_counters once per column to
// restart the face from the correct horizontal offset.
//
//   for (col = 0; col < out_w; ++col) {                             // SW: output cols only
//     src_base = input + col * elem;
//     reset_counters();                                             // restart face for this col
//     for (row = 0; row < out_h; ++row) {                          // HW face: output rows
//       for (outer = 0; outer < k*row_bytes; outer += row_bytes) { // HW outer: kernel rows
//         for (inner = 0; inner < k*elem; inner += elem) {          // HW inner: kernel cols
//           yield src_base + face_offset + outer + inner
//         }
//       }
//       // face_offset += face_stride automatically
//     }
//   }
//
// This eliminates one SW loop level compared to the general im2col example, which is
// beneficial for large images where out_h is big.
//
// Compile-time args:
//   0: src_stride_en    - unused
//   1: dst_stride_en    - unused
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

// Kernel: 3×3, dilation=1, convolution stride=1
constexpr uint32_t k = 3;
constexpr uint32_t s = 1;

// Output dimensions
constexpr uint32_t out_w = img_width - (k - 1);   // 6
constexpr uint32_t out_h = img_height - (k - 1);  // 2

constexpr uint32_t src_base = 0x10000;
constexpr uint32_t dst_base = 0x200000;

// Face loop advances the source base by one output row after each k*k patch
constexpr uint64_t face_stride = (uint64_t)s * row_bytes;  // 64

// HW inner: kernel columns (fast axis)
constexpr uint64_t src_inner_stride = elem;             // 8
constexpr uint64_t src_inner_end = (uint64_t)k * elem;  // 24

// HW outer: kernel rows (slow axis)
constexpr uint64_t src_outer_stride = row_bytes;             // 64
constexpr uint64_t src_outer_end = (uint64_t)k * row_bytes;  // 192

// HW dest: linear output
constexpr uint32_t total_addresses = out_h * out_w * k * k;  // 108
constexpr uint64_t dst_inner_stride = elem;
constexpr uint64_t dst_inner_end = (uint64_t)total_addresses * elem + 1;

void kernel_main() {
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    // Face loop: after each k*k patch the HW advances base by face_stride (one output row)
    setup_src_face_size_addrgen_0(face_stride);
    setup_src_inner_loop_addrgen_0(src_inner_stride, src_inner_end);
    setup_src_outer_loop_addrgen_0(src_outer_stride, src_outer_end);

    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(dst_inner_stride, dst_inner_end);

    /* For real NOC transfers peek/pop are not needed — replace this loop body with:
     *   push_both_addrgen_0();
     *   issue_transaction_cmdbuf_0; */
    for (uint32_t col = 0; col < out_w; ++col) {
        // Point the face start at the left edge of this output column
        setup_src_base_start_addrgen_0(src_base + (uint64_t)col * s * elem);
        // Reset face/inner/outer counters so the face restarts from row 0 for this column
        reset_counters_addrgen_0();

        // The face loop auto-advances through out_h output rows; no SW row loop needed
        for (uint32_t row = 0; row < out_h; ++row) {
            for (uint32_t p = 0; p < k * k; ++p) {
                uint64_t src_addr = peek_src_addrgen_0();
                uint64_t dest_addr = peek_dest_addrgen_0();
                pop_src_addrgen_0();
                pop_dest_addrgen_0();
                DPRINT << "  [out_col=" << col << " out_row=" << row << " patch=" << p << "]"
                       << " Src: " << HEX() << (uint32_t)src_addr << " Dst: " << HEX() << (uint32_t)dest_addr << ENDL();
            }
        }
    }
}
