// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47c fused MTP RMSNorm, compute. xn = x (+ residual) in fp32 (SFPU add, one rounding); sum of squares per core = fp32
// SFPU tile fold + SFPU row reduce (like the stock accurate-fp32 ttnn.sum); (exchange of the T partials in the reader);
// rs = rsqrt(SFPU row reduce of R + eps'); out_g = bf16((xn_g * rs) * w_g) with fp32 SFPU multiplies (as the stock
// _rms6 chain: multiply, multiply(dtype=bf16)). Fp32 tiles read by copy_tile are unpacked straight into dest (host sets
// UnpackToDestFp32).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(3);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(4);
    constexpr uint32_t cb_p = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rsb = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t GPC = get_compile_time_arg_val(10);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(11);
    constexpr uint32_t HAS_R = get_compile_time_arg_val(12);
    constexpr uint32_t OUT_X = get_compile_time_arg_val(13);
    constexpr uint32_t cb_s = HAS_R ? cb_xn : cb_x;  // the (summed) x the rest reads

    compute_kernel_hw_startup(cb_x, cb_w, cb_p);
    CircularBuffer x(cb_x), r(cb_r), w(cb_w), xn(cb_xn), xo(cb_xo), p(cb_p), tot(cb_tot), rs(cb_rs), rsb(cb_rsb),
        out(cb_out);
    x.wait_front(GPC);

    if constexpr (HAS_R) {
        r.wait_front(GPC);
        pack_reconfig_data_format(cb_xn);
        for (uint32_t g = 0; g < GPC; ++g) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            copy_tile(cb_x, g, 0);
            reconfig_data_format_srca(cb_r);
            copy_init(cb_r);
            copy_tile(cb_r, g, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            xn.reserve_back(1);
            pack_tile(0, cb_xn);
            xn.push_back(1);
            if constexpr (OUT_X) {
                xo.reserve_back(1);
                pack_tile(0, cb_xo);
                xo.push_back(1);
            }
            tile_regs_release();
        }
        xn.wait_front(GPC);
    }

    // per-core sum of squares: dest0 = sum_g x_g^2 (fp32 SFPU, tile order), then the SFPU row reduce -> column 0
    pack_reconfig_data_format(cb_p);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_s);
    copy_init(cb_s);
    for (uint32_t g = 0; g < GPC; ++g) {
        if (g == 0) {
            copy_tile(cb_s, 0, 0);
            copy_tile(cb_s, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
        } else {
            copy_tile(cb_s, g, 1);
            copy_tile(cb_s, g, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    p.reserve_back(1);
    pack_tile(0, cb_p);
    p.push_back(1);
    tile_regs_release();

    // rs = rsqrt(sum_k R[:, k] + eps')   (R[t, k] = partial sum of squares of core k, token t)
    tot.wait_front(1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_tot);
    copy_init(cb_tot);
    copy_tile(cb_tot, 0, 0);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0, 1, 1);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    rsb.wait_front(1);  // the reader broadcasts column 0 to every column
    w.wait_front(GPC);

    pack_reconfig_data_format(cb_out);
    for (uint32_t g = 0; g < GPC; ++g) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, g, 0);
        reconfig_data_format_srca(cb_rsb);
        copy_init(cb_rsb);
        copy_tile(cb_rsb, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_w);
        copy_init(cb_w);
        copy_tile(cb_w, g, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
    }
}
