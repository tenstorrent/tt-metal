// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused anti-aliased SnakeBeta compute kernel.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ab = get_compile_time_arg_val(2);
    constexpr uint32_t cb_e = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dn = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t NB_EXTRA = get_compile_time_arg_val(15);

    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    if (n_tiles == 0) {
        return;
    }
    const uint32_t nblocks = n_tiles + NB_EXTRA;
    uint32_t s[12];
    uint32_t t[12];
    for (uint32_t k = 0; k < 12; ++k) {
        s[k] = get_common_arg_val<uint32_t>(k);
        t[k] = get_common_arg_val<uint32_t>(12 + k);
    }

    CircularBuffer up(cb_up);
    CircularBuffer ab(cb_ab);
    CircularBuffer e(cb_e);
    CircularBuffer o(cb_o);
    CircularBuffer dn(cb_dn);
    CircularBuffer out(cb_out);

    constexpr uint32_t DST_E = 0, DST_O = 1, DST_T = 2, DST_A = 3, DST_B = 4, DST_E2 = 5, DST_O2 = 6;

    compute_kernel_hw_startup(cb_up, cb_out);
    binop_with_scalar_tile_init();
    add_binary_tile_init();
    snake_beta_tile_init();

    ab.wait_front(2);

    pack_reconfig_data_format(cb_e);
    for (uint32_t blk = 0; blk < nblocks; ++blk) {
        up.wait_front(7);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ab);
        copy_init(cb_ab);
        copy_tile(cb_ab, 0, DST_A);
        copy_tile(cb_ab, 1, DST_B);
        reconfig_data_format_srca(cb_up);
        copy_init(cb_up);
        copy_tile(cb_up, 0, DST_E);
        mul_unary_tile(DST_E, s[0]);
        for (uint32_t j = 1; j < 6; ++j) {
            copy_tile(cb_up, j, DST_T);
            mul_unary_tile(DST_T, s[2 * j]);
            add_binary_tile(DST_E, DST_T, DST_E);
        }
        copy_tile(cb_up, 1, DST_O);
        mul_unary_tile(DST_O, s[1]);
        for (uint32_t j = 1; j < 6; ++j) {
            copy_tile(cb_up, 1 + j, DST_T);
            mul_unary_tile(DST_T, s[2 * j + 1]);
            add_binary_tile(DST_O, DST_T, DST_O);
        }
        snake_beta_tile<DataFormat::Float32>(DST_E, DST_A, DST_B, DST_E2);
        snake_beta_tile<DataFormat::Float32>(DST_O, DST_A, DST_B, DST_O2);
        tile_regs_commit();
        tile_regs_wait();
        e.reserve_back(1);
        o.reserve_back(1);
        pack_tile(DST_E2, cb_e);
        pack_reconfig_data_format(cb_o);
        pack_tile(DST_O2, cb_o);
        pack_reconfig_data_format(cb_e);
        e.push_back(1);
        o.push_back(1);
        tile_regs_release();
        up.pop_front(7);
    }

    pack_reconfig_data_format(cb_out);
    reconfig_data_format_srca(cb_dn);
    copy_init(cb_dn);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        dn.wait_front(12);
        tile_regs_acquire();
        copy_tile(cb_dn, 0, DST_E);
        mul_unary_tile(DST_E, t[0]);
        for (uint32_t k = 1; k < 12; ++k) {
            copy_tile(cb_dn, k, DST_T);
            mul_unary_tile(DST_T, t[k]);
            add_binary_tile(DST_E, DST_T, DST_E);
        }
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(DST_E, cb_out);
        out.push_back(1);
        tile_regs_release();
        dn.pop_front(12);
    }
}
