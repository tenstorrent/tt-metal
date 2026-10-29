// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Post-SDPA glue rope compute.  Fused (== rope_compute.cpp): out = x @ R.  UNF (the stock unfused addcmul formula of
// spec_attention._finish, T > 8 and T % 8 != 0): out = addcmul(x * c, x @ P, s) with the SFPU ops of the stock ternary
// / binary kernels.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr bool UNF = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_r2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ri = get_compile_time_arg_val(5);
    constexpr uint32_t cb_m = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    compute_kernel_hw_startup(cb_x, cb_r, cb_out);
    CircularBuffer x(cb_x), r(cb_r), out(cb_out);
    x.wait_front(1);
    r.wait_front(1);
    reconfig_data_format(cb_x, cb_r);
    matmul_init(cb_x, cb_r);
    if constexpr (!UNF) {
        tile_regs_acquire();
        matmul_tiles(cb_x, cb_r, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    } else {
        // stock: p = x @ P (exact), m = SFPU x * c, out = SFPU addcmul(m, p, s), all in bf16 DEST: the arithmetic of
        // ttnn.addcmul(ttnn.multiply(x, c), x @ Pf, s)
        CircularBuffer r2(cb_r2), sfull(cb_ri), p(cb_p);
        r2.wait_front(1);
        sfull.wait_front(1);
        matmul_init(cb_x, cb_r2);
        tile_regs_acquire();
        matmul_tiles(cb_x, cb_r2, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        p.reserve_back(1);
        pack_tile(0, cb_p);
        p.push_back(1);
        tile_regs_release();
        p.wait_front(1);
        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        reconfig_data_format_srca(cb_x, cb_r);
        copy_init(cb_r);
        copy_tile(cb_r, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // m = x * c (SFPU, rounded to bf16 in DEST like the stock multiply)
        reconfig_data_format_srca(cb_r, cb_p);
        copy_init(cb_p);
        copy_tile(cb_p, 0, 1);
        reconfig_data_format_srca(cb_p, cb_ri);
        copy_init(cb_ri);
        copy_tile(cb_ri, 0, 2);
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, 0x3F800000u);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
}
