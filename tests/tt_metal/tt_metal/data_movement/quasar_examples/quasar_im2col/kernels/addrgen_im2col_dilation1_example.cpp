// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Im2col for dilation = 1, direct port of im2col_dilation_1.cpp to the addrgen API.
//
// The address generator is configured ONCE before the loop:
//   face:  advances src_base by one row (row_bytes) after each complete inner×outer cycle
//   inner: sweeps kernel rows    (stride = row_bytes,  end = row_bytes*(kH-1) + 1)
//   outer: sweeps kernel columns (stride = elem_size,  end = row_bytes - kW*elem_size + 1)
//
// One complete inner×outer cycle = kH * kW addresses = one kernel patch.
// After every kH*kW pops the face automatically advances by row_bytes (next output row).
//
// SW loop structure (identical in spirit to the original):
//   for i in [0, matrix_size):     // one iteration per image pixel
//     for j in [0, kH*kW):         // one element per kernel position
//       peek / pop
//
// No reset_counters between iterations — the face auto-advances.
//
// Image: H x W = 4 x 5, elem_size = 128 B
// Kernel: kH x kW = 3 x 3
// matrix_size = H * W = 20
// Total addresses = matrix_size * kH * kW = 180
//
// Compile-time args:
//   0: (unused)
//   1: (unused)
//   2: num_of_addresses — must equal matrix_size * kH * kW

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include <cstdint>

constexpr uint32_t H = 4;
constexpr uint32_t W = 5;
constexpr uint32_t kH = 3;
constexpr uint32_t kW = 3;
constexpr uint32_t elem_size = 128;
constexpr uint32_t row_bytes = W * elem_size;  // 640

constexpr uint32_t src_base = 0x10000;
constexpr uint32_t dst_base = 0x20000;

// Face: one image row per kernel patch
constexpr uint64_t face_stride = row_bytes;

// Inner: kernel rows  (matches original inner_stride / inner_loop_end)
constexpr uint64_t inner_stride = row_bytes;
constexpr uint64_t inner_end = (uint64_t)row_bytes * (kH - 1) + 1;  // 1281

// Outer: kernel columns  (matches original outer_stride / outer_loop_end)
constexpr uint64_t outer_stride = elem_size;
constexpr uint64_t outer_end = (uint64_t)row_bytes - (uint64_t)kW * elem_size + 1;  // 257

constexpr uint32_t matrix_size = H * W;                      // 20
constexpr uint32_t total_addresses = matrix_size * kH * kW;  // 180

void kernel_main() {
    constexpr uint32_t num_of_addresses = get_compile_time_arg_val(2);

    reset_addrgen_0();

    // Configure once — face/inner/outer are never touched inside the loop
    setup_src_base_start_addrgen_0(src_base);
    setup_src_face_size_addrgen_0(face_stride);
    setup_src_inner_loop_addrgen_0(inner_stride, inner_end);
    setup_src_outer_loop_addrgen_0(outer_stride, outer_end);

    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(elem_size, (uint64_t)total_addresses * elem_size);

    /* For real NOC transfers peek/pop are not needed — replace this loop body with:
     *   push_both_addrgen_0();
     *   issue_transaction_cmdbuf_0; */
    for (uint32_t i = 0; i < matrix_size; ++i) {
        for (uint32_t j = 0; j < kH * kW; ++j) {
            uint64_t src_addr = peek_src_addrgen_0();
            uint64_t dest_addr = peek_dest_addrgen_0();
            pop_src_addrgen_0();
            pop_dest_addrgen_0();
            DPRINT << "  Source address: " << HEX() << (uint64_t)src_addr << " Destination address: " << HEX()
                   << (uint64_t)dest_addr << ENDL();
        }
    }
}
