// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for RmsNormDecode: x^2 of this core's `per` tiles (FPU, fp32) -> cb_sq; their row sums (FPU reduce with
// the unit scaler) -> the fp32 partial tile cb_part (column 0); after the exchange, the N partials summed on dest,
// inv = rsqrt(sum / W + eps) * scalar (SFPU) -> cb_inv; then every own tile x * inv (column broadcast) * gamma
// (row broadcast) on fp32 dest, one bf16 pack into cb_out.
// compile-time args: [cb_x, cb_g, cb_scaler, cb_sq, cb_part, cb_recv, cb_inv, cb_out, per, N, has_gamma, eps_bits,
//                     inv_w_bits, scalar_bits]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_G = get_compile_time_arg_val(1);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(2);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(3);
constexpr uint32_t CB_PART = get_compile_time_arg_val(4);
constexpr uint32_t CB_RECV = get_compile_time_arg_val(5);
constexpr uint32_t CB_INV = get_compile_time_arg_val(6);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(7);
constexpr uint32_t PER = get_compile_time_arg_val(8);
constexpr uint32_t N = get_compile_time_arg_val(9);
constexpr bool HAS_GAMMA = get_compile_time_arg_val(10) == 1;
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(11);
constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(12);
constexpr uint32_t SCALAR_BITS = get_compile_time_arg_val(13);

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_SCALER, CB_OUT);
    CircularBuffer cx(CB_X), cg(CB_G), cs(CB_SCALER), csq(CB_SQ), cp(CB_PART), cr(CB_RECV), ci(CB_INV), co(CB_OUT);
    cx.wait_front(PER);
    // x^2 tiles (fp32)
    reconfig_data_format(CB_X, CB_X);
    mul_init(CB_X, CB_X);
    pack_reconfig_data_format(CB_SQ);
    csq.reserve_back(PER);
    for (uint32_t c = 0; c < PER; ++c) {
        tile_regs_acquire();
        mul_tiles(CB_X, CB_X, c, c, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SQ, c);
        tile_regs_release();
    }
    csq.push_back(PER);
    csq.wait_front(PER);
    cs.wait_front(1);
    // partial row sums -> cb_part
    reconfig_data_format(CB_SCALER, CB_SQ);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_PART);
    tile_regs_acquire();
    for (uint32_t c = 0; c < PER; ++c) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, c, 0, 0);
    }
    reduce_uninit(CB_SQ);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_PART);
    cp.reserve_back(1);
    pack_tile(0, CB_PART, 0);
    cp.push_back(1);
    tile_regs_release();
    csq.pop_front(PER);
    // the N partials -> inv
    cr.wait_front(N);
    reconfig_data_format_srca(CB_RECV);
    copy_init(CB_RECV);
    tile_regs_acquire();
    copy_tile(CB_RECV, 0, 0);
    add_binary_tile_init();
    for (uint32_t j = 1; j < N; ++j) {
        copy_tile(CB_RECV, j, 1);
        add_binary_tile(0, 1, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_W_BITS);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile<RsqrtMode::Default>(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, SCALAR_BITS);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_INV);
    ci.reserve_back(1);
    pack_tile(0, CB_INV, 0);
    ci.push_back(1);
    tile_regs_release();
    cr.pop_front(N);
    ci.wait_front(1);
    if constexpr (HAS_GAMMA) {
        cg.wait_front(PER);
    }
    // normalize own tiles
    pack_reconfig_data_format(CB_OUT);
    co.reserve_back(PER);
    for (uint32_t c = 0; c < PER; ++c) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_X);
        copy_init(CB_X);
        copy_tile(CB_X, c, 0);
        reconfig_data_format_srca(CB_INV);
        copy_init(CB_INV);
        copy_tile(CB_INV, 0, 1);
        if constexpr (HAS_GAMMA) {  // every unpack copy before the first SFPU op of the window
            reconfig_data_format_srca(CB_G);
            copy_init(CB_G);
            copy_tile(CB_G, c, 2);
        }
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        if constexpr (HAS_GAMMA) {
            sfpu_bcast_row_init();
            sfpu_mul_bcast_row(0, 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT, c);
        tile_regs_release();
    }
    co.push_back(PER);
    ci.pop_front(1);
    cx.pop_front(PER);
    if constexpr (HAS_GAMMA) {
        cg.pop_front(PER);
    }
}
