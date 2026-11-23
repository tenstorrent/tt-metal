// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Near partial combine, compute (fp32 SFPU, full tiles, 8 fp32 DST tiles: dst_full_sync_en; the c_i are row
// maxima broadcast over the columns):
//   M = max_i c_i,  s_i = exp(c_i - M),  z = sum_i s_i z_i,  o = (sum_i s_i O_i) / z -> cb_out (fp32);
//   lse = M + log z, sink tile = bf16(lse) (cb_st), r = exp(bf16(lse) - lse) (cb_r).
#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_z = 0, cb_o = 1, cb_c = 2, cb_zs = 16, cb_out = 17, cb_lf = 18, cb_st = 19, cb_r = 20;
constexpr uint32_t cb_m = 21, cb_s = 22;
constexpr uint32_t dt = 4;
// SFPU ops on the top two faces only (rows 0..15; the group's heads are rows 0..2, every op is per element)
constexpr VectorMode VM = VectorMode::R;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
// x_i = cb[first + i * stride] in DST i, s_i in DST n + i: DST 0 = ((s_0 x_0 + s_1 x_1) + s_2 x_2) + ..., the same
// products and sums, in the same order, as one tile at a time; then (divide) / z -> out
FORCE_INLINE void wsum(uint32_t cb, uint32_t first, uint32_t stride, uint32_t n, bool divide, uint32_t out) {
    tile_regs_acquire();
    reconfig_data_format_srca(cb);
    copy_init(cb);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb, first + i * stride, i);
    }
    reconfig_data_format_srca(cb_s);
    copy_init(cb_s);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_s, i, n + i);
    }
    mul_binary_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        mul_binary_tile(i, n + i, i, VM);
    }
    add_binary_tile_init();
    for (uint32_t i = 1; i < n; ++i) {
        add_binary_tile(0, i, 0, VM);
    }
    if (divide) {
        load(cb_zs, 0, 1);
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    put(0, out);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t nsplit = get_named_compile_time_arg_val("nsplit");
    compute_kernel_hw_startup(cb_z, cb_out);
    cb_wait_front(cb_z, nsplit);
    cb_wait_front(cb_o, nsplit * dt);
    cb_wait_front(cb_c, nsplit);

    // M and the partial scales s_i in one DST acquire (8 fp32 DST tiles, full sync): c_i in DST i, M in DST n.
    // Same ops on the same values as one tile at a time: M = max(max(max(c_0, c_1), c_2), c_3), s_i = exp(c_i - M)
    static_assert(2 * nsplit <= 8, "c_i, s_i and M must fit the 8 fp32 DST tiles");
    tile_regs_acquire();
    reconfig_data_format_srca(cb_c);
    copy_init(cb_c);
    for (uint32_t i = 0; i < nsplit; ++i) {
        copy_tile(cb_c, i, i);
    }
    copy_tile(cb_c, 0, nsplit);
    binary_max_tile_init();
    for (uint32_t i = 1; i < nsplit; ++i) {
        binary_max_tile(nsplit, i, nsplit, VM);
    }
    sub_binary_tile_init();
    for (uint32_t i = 0; i < nsplit; ++i) {
        sub_binary_tile(i, nsplit, i, VM);
    }
    exp_tile_init();
    for (uint32_t i = 0; i < nsplit; ++i) {
        exp_tile(i, VM);
    }
    tile_regs_commit();
    tile_regs_wait();
    put(nsplit, cb_m);
    for (uint32_t i = 0; i < nsplit; ++i) {
        put(i, cb_s);
    }
    tile_regs_release();
    cb_wait_front(cb_m, 1);
    cb_wait_front(cb_s, nsplit);

    // z, then the output tiles: sum_i s_i x_i (/ z), 2 nsplit DST tiles per acquire
    wsum(cb_z, 0, 1, nsplit, false, cb_zs);
    cb_wait_front(cb_zs, 1);
    for (uint32_t t = 0; t < dt; ++t) {
        wsum(cb_o, t, dt, nsplit, true, cb_out);
    }

    {
        tile_regs_acquire();
        load(cb_m, 0, 0);
        load(cb_zs, 0, 1);
        log_tile_init();
        log_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0, VM);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_lf);
        put(0, cb_st);
        tile_regs_release();
        cb_wait_front(cb_lf, 1);
        cb_wait_front(cb_st, 1);
        tile_regs_acquire();
        load(cb_st, 0, 0);
        load(cb_lf, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0, VM);
        exp_tile_init();
        exp_tile(0, VM);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_r);
        tile_regs_release();
    }
}
