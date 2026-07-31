// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    // Compile time args
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    // Runtime args -- which scalar op family to run, and the raw param1 bits to hand it.
    // op_mode: 0=div_unary_tile 1=mul_unary_tile 2=add_unary_tile 3=sub_unary_tile 4=rsub_unary_tile
    uint32_t op_mode = get_arg_val<uint32_t>(0);
    uint32_t scalar_bits = get_arg_val<uint32_t>(1);

    constexpr uint32_t one_tile = 1;

    init_sfpu(cb_in0, cb_out);

    tile_regs_acquire();

    cb_wait_front(cb_in0, one_tile);
    cb_wait_front(cb_in1, one_tile);
    copy_tile(cb_in0, /*offset*/ 0, /*register_offset*/ 0);
    copy_tile(cb_in1, /*offset*/ 0, /*register_offset*/ 1);

    // Mirrors the customer's exact pattern: one SFPU binary op, then one SFPU scalar op, both
    // inside the same tile_regs_acquire()/commit() block, both applied to DST[0].
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // DST[0] = in0 + in1 = 1.0 + 1.0 = 2.0

    binop_with_scalar_tile_init();  // required: switching SFPU op family from eltwise_binary_sfpu
                                    // to binop_with_scalar.
    if (op_mode == 0) {
        div_unary_tile(0, scalar_bits);
    } else if (op_mode == 1) {
        mul_unary_tile(0, scalar_bits);
    } else if (op_mode == 2) {
        add_unary_tile(0, scalar_bits);
    } else if (op_mode == 3) {
        sub_unary_tile(0, scalar_bits);
    } else {
        rsub_unary_tile(0, scalar_bits);
    }

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, one_tile);
    pack_tile(0, cb_out);

    cb_pop_front(cb_in0, one_tile);
    cb_pop_front(cb_in1, one_tile);

    tile_regs_release();

    cb_push_back(cb_out, one_tile);
}
