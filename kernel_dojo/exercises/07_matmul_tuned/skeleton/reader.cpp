// SPDX-License-Identifier: Apache-2.0
//
// Exercise 07 — matmul reader with A-row reuse.
//
// This core owns rows [start_row, start_row + n_rows) of C. For each row:
//   - push A's row (Kt tiles) into cb_a ONCE
//   - then push each of B's Nt columns (Kt tiles each) into cb_b
//
// The compute kernel keeps A's row resident for the whole row, so A is read
// Mt*Kt times total instead of Mt*Nt*Kt.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Kt = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);
    const uint32_t start_row = get_arg_val<uint32_t>(4);
    const uint32_t n_rows = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);

    constexpr auto a_args = TensorAccessorArgs<2>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const uint32_t end_row = start_row + n_rows;

    for (uint32_t mt = start_row; mt < end_row; mt++) {
        // TODO: reserve Kt pages in cb_a, read A tiles (mt, 0..Kt-1) —
        //       page index mt * Kt + kt — barrier, push Kt.

        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: reserve Kt pages in cb_b, read B tiles (0..Kt-1, nt) —
            //       page index kt * Nt + nt — barrier, push Kt.
        }
    }
}
