// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Global-layer near softmax, compute, on one dense row (page A: 1024 near scores, page B: the sink block row,
// -inf elsewhere), the math of attention_fast._global:
//   x = near + bias; m = max(x, sink); e = exp(. - m); z = sum(e) (two HiFi4 matmuls with a ones tile, which
//   also broadcast z to every element); p = e_A * (1 / z) -> bf16; lse = m + log(z).
// m only stabilizes the exps (lse uses the same broadcast m), so the FPU max reduce precision does not matter.
#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, 0, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    // to-DEST fp32: a, b, xa, m, ea, z, mb; srcA/srcB: bias (bf16), ones, scaler, mx, s, rs
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_bias = 2, cb_ones = 3, cb_scaler = 4;
    constexpr uint32_t cb_xa = 16, cb_mx = 17, cb_m = 18, cb_ea = 19, cb_s = 20, cb_mb = 21, cb_rs = 22;
    constexpr uint32_t cb_z = 23, cb_p = 28, cb_lse = 29;

    compute_kernel_hw_startup(cb_a, cb_bias, cb_xa);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_wait_front(cb_bias, 1);
    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_scaler, 1);

    // x = near + bias; elementwise max(x, sink page) for the max reduce
    tile_regs_acquire();
    load(cb_a, 0);
    load(cb_bias, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    load(cb_b, 2);
    binary_max_tile_init();
    binary_max_tile(0, 2, 3);
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_xa);
    put(3, cb_mx);
    tile_regs_release();

    // m = max over the page (FPU reduce, scalar at [0, 0])
    cb_wait_front(cb_mx, 1);
    reconfig_data_format(cb_mx, cb_scaler);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_mx, cb_scaler, cb_m);
    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR>(cb_mx, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    reduce_uninit();
    tile_regs_wait();
    put(0, cb_m);
    tile_regs_release();
    cb_pop_front(cb_mx, 1);

    // e_A = exp(x - m), e_B = exp(sink - m), s = e_A + e_B; keep the broadcast m for lse
    cb_wait_front(cb_xa, 1);
    cb_wait_front(cb_m, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_m);
    unary_bcast_init<BroadcastType::SCALAR>(cb_m);
    unary_bcast<BroadcastType::SCALAR>(cb_m, 0, 1);
    unary_bcast_uninit<BroadcastType::SCALAR>(cb_m);
    load(cb_xa, 0);
    load(cb_b, 2);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    sub_binary_tile(2, 1, 2);
    exp_tile_init();
    exp_tile(0);
    exp_tile(2);
    add_binary_tile_init();
    add_binary_tile(0, 2, 3);
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_ea);
    put(3, cb_s);
    put(1, cb_mb);
    tile_regs_release();
    cb_pop_front(cb_xa, 1);
    cb_pop_front(cb_m, 1);

    // z = ones x (s x ones): row sums, then the column sum of those -> the total in every element
    cb_wait_front(cb_s, 1);
    reconfig_data_format(cb_ones, cb_s);  // matmul: in1 -> SrcA, in0 -> SrcB
    matmul_init(cb_s, cb_ones);
    tile_regs_acquire();
    matmul_tiles(cb_s, cb_ones, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_rs);
    tile_regs_release();
    cb_pop_front(cb_s, 1);
    cb_wait_front(cb_rs, 1);
    reconfig_data_format(cb_rs, cb_ones);
    matmul_init(cb_ones, cb_rs);
    tile_regs_acquire();
    matmul_tiles(cb_ones, cb_rs, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_z);
    tile_regs_release();
    cb_pop_front(cb_rs, 1);

    // p = e_A * (1 / z) (bf16); lse = m + log(z)
    cb_wait_front(cb_z, 1);
    cb_wait_front(cb_ea, 1);
    cb_wait_front(cb_mb, 1);
    tile_regs_acquire();
    load(cb_z, 0);
    recip_tile_init();
    recip_tile(0);
    load(cb_ea, 1);
    mul_binary_tile_init();
    mul_binary_tile(1, 0, 1);
    load(cb_z, 2);
    log_tile_init();
    log_tile(2);
    load(cb_mb, 3);
    add_binary_tile_init();
    add_binary_tile(2, 3, 2);
    tile_regs_commit();
    tile_regs_wait();
    put(1, cb_p);
    put(2, cb_lse);
    tile_regs_release();
    cb_pop_front(cb_z, 1);
    cb_pop_front(cb_ea, 1);
    cb_pop_front(cb_mb, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    cb_pop_front(cb_bias, 1);
    cb_pop_front(cb_ones, 1);
    cb_pop_front(cb_scaler, 1);
}
