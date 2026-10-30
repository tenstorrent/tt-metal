// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rmsnorm_bw_rows compute (one item (row r, slice s) with the slices resident in L1):
//   pass 1: partial = sum over the slice of a * gamma * dy  (fp32 DEST)  -> exchanged between the row's cores
//   inv = 1 / rms, t = rowsum(sum_s partial_s) * inv^2 / C
//   pass 2: dL_da = inv * (gamma * dy - t * a) [+ residual_grad]

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/kernels/rmsnorm_bw_rows_cbs.hpp"

namespace cb = rmsnorm_bw_rows_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t inv_c_bits = get_compile_time_arg_val(4);

FORCE_INLINE void load_tile(uint32_t cb_id, uint32_t idx, uint32_t reg) {
    reconfig_data_format_srca(cb_id);
    copy_init(cb_id);
    copy_tile(cb_id, idx, reg);
}

FORCE_INLINE void gamma_times_dy(uint32_t idx, uint32_t reg) {
    load_tile(cb::zero, 0, reg);
    reconfig_data_format(cb::dy, cb::gamma);
    mul_bcast_rows_init(cb::dy, cb::gamma);
    mul_tiles_bcast_rows(cb::dy, cb::gamma, idx, idx, reg);
}

FORCE_INLINE void matmul_with_constant(uint32_t cb_x, uint32_t cb_k, uint32_t reg) {
    reconfig_data_format(cb_k, cb_x);
    matmul_init(cb_x, cb_k, 0);
    matmul_tiles(cb_x, cb_k, 0, 0, reg);
}

void kernel_main() {
    [[maybe_unused]] const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(1);
    const uint32_t col0 = s * St;
    const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);

    compute_kernel_hw_startup(cb::dy, cb::gamma, cb::dx);
    cb_wait_front(cb::zero, 1);
    cb_wait_front(cb::ones, 1);
    cb_wait_front(cb::ones_row0, 1);
    cb_wait_front(cb::a, St);
    cb_wait_front(cb::gamma, St);
    cb_wait_front(cb::dy, St);

    // pass 1: the slice's row dot product
    tile_regs_acquire();
    load_tile(cb::zero, 0, 0);
    for (uint32_t i = 0; i < ncols; ++i) {
        gamma_times_dy(i, 1);
        load_tile(cb::a, i, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    pack_and_push(0, cb::partial_out);

    // inv = 1 / rms (column 0 of the rms tile broadcast along columns)
    cb_wait_front(cb::rms, 1);
    tile_regs_acquire();
    matmul_with_constant(cb::rms, cb::ones_row0, 0);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    pack_and_push(0, cb::inv);

    // acc = sum of the row's partials (slot order = slice order on every core: the same summation everywhere)
    cb_wait_front(cb::partials, S);
    tile_regs_acquire();
    load_tile(cb::partials, 0, 0);
    add_binary_tile_init();
    for (uint32_t j = 1; j < S; ++j) {
        load_tile(cb::partials, j, 1);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    pack_and_push(0, cb::acc);

    // t = rowsum(acc) * inv * inv / C
    cb_wait_front(cb::acc, 1);
    cb_wait_front(cb::inv, 1);
    tile_regs_acquire();
    matmul_with_constant(cb::acc, cb::ones, 0);
    load_tile(cb::inv, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    mul_binary_tile(0, 1, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_c_bits);
    tile_regs_commit();
    pack_and_push(0, cb::t);
    cb_pop_front(cb::acc, 1);
    cb_wait_front(cb::t, 1);

    // pass 2: dL_da = inv * (gamma * dy - t * a) [+ dres], in full blocks for the writer
    for (uint32_t c = 0; c < ncols; c += block) {
        const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
#ifdef ADD_RESIDUAL_GRAD
        cb_wait_front(cb::dres, block);
#endif
        for (uint32_t i = 0; i < n; ++i) {
            tile_regs_acquire();
            gamma_times_dy(c + i, 0);
            load_tile(cb::t, 0, 1);
            load_tile(cb::a, c + i, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            load_tile(cb::inv, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
#ifdef ADD_RESIDUAL_GRAD
            load_tile(cb::dres, i, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
#endif
            tile_regs_commit();
            pack_and_push(0, cb::dx);
        }
        for (uint32_t i = n; i < block; ++i) {  // tail padding (unused tiles)
            tile_regs_acquire();
            tile_regs_commit();
            pack_and_push(0, cb::dx);
        }
#ifdef ADD_RESIDUAL_GRAD
        cb_pop_front(cb::dres, block);
#endif
    }

    cb_pop_front(cb::rms, 1);
    cb_pop_front(cb::partials, S);
    cb_pop_front(cb::inv, 1);
    cb_pop_front(cb::t, 1);
    cb_pop_front(cb::a, St);
    cb_pop_front(cb::gamma, St);
    cb_pop_front(cb::dy, St);
}
