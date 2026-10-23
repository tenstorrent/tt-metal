// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for RopeDecode. Per work unit (one row tile, the column pair a = c, b = c + half_t):
//   out_a = x_a * cos_a - x_b * sin_a        (rotate_half: the first half pairs with -x2)
//   out_b = x_b * cos_b + x_a * sin_b        (the second half pairs with +x1)
// cos / sin tiles hold the user's row in row 0 and are row-broadcast by the FPU (mul_tiles_bcast_rows); the two
// products meet on dest (fp32) through the binary SFPU sub / add, so no sign tile or rotated copy is built.
//
// compile-time args: [cb_x, cb_cos, cb_sin, cb_out, mode]
//   mode 2 is the op; 0 (out = x) and 1 (out = x * cos) are the isolation ladder of the unit test.
// runtime args: [unit count]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_COS = get_compile_time_arg_val(1);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(2);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(3);
constexpr uint32_t MODE = get_compile_time_arg_val(4);

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_COS, CB_OUT);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer cbx(CB_X), cbc(CB_COS), cbs(CB_SIN), cbo(CB_OUT);
    constexpr uint32_t D_A = 0, D_B = 1, D_SA = 2, D_SB = 3;  // out_a, out_b, x_b*sin_a, x_a*sin_b
    for (uint32_t t = 0; t < count; ++t) {
        cbx.wait_front(2);
        cbc.wait_front(2);
        cbs.wait_front(2);
        tile_regs_acquire();
        if constexpr (MODE == 0) {
            copy_init(CB_X);
            copy_tile(CB_X, 0, D_A);
            copy_tile(CB_X, 1, D_B);
        } else {
            mul_bcast_rows_init(CB_X, CB_COS);
            mul_tiles_bcast_rows(CB_X, CB_COS, 0, 0, D_A);
            mul_tiles_bcast_rows(CB_X, CB_COS, 1, 1, D_B);
            if constexpr (MODE == 2) {
                mul_bcast_rows_init(CB_X, CB_SIN);
                mul_tiles_bcast_rows(CB_X, CB_SIN, 1, 0, D_SA);
                mul_tiles_bcast_rows(CB_X, CB_SIN, 0, 1, D_SB);
                sub_binary_tile_init();
                sub_binary_tile(D_A, D_SA, D_A);
                add_binary_tile_init();
                add_binary_tile(D_B, D_SB, D_B);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cbo.reserve_back(2);
        pack_tile(D_A, CB_OUT, 0);
        pack_tile(D_B, CB_OUT, 1);
        cbo.push_back(2);
        tile_regs_release();
        cbx.pop_front(2);
        cbc.pop_front(2);
        cbs.pop_front(2);
    }
}
