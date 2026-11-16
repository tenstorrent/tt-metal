// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gemma decode layer boundary, compute (TRISC) (fused_decode/boundary.py: GemmaBoundary).
//
// Every vector is flat (value h at element h of `tiles` 32x32 pages, zero padded), so every op is elementwise or a
// whole-vector sum and the tile structure is only a container (models/demos/gpt_oss fused_decode boundary contract).
// norm(v, g) = v * g * rsqrt(sum(v^2) / hidden + eps), statistics in FP32.
//
// mode 0 (entry):  x = norm(res, g0)                                         -> out0 (BF16)
// mode 1 (attn):   a = sum of the slots (o_proj partials, slot order)
//                  res' = res + norm(a, g0)                                  -> cb_res_out (BF16)
//                  n = norm(res', 1): out0 = n * vf (FP32, router input), out1 = n * g1, out2 = n * g2 (BF16)
// mode 2 (moe):    s, r = sum of the slots of part 0 (shared MLP) / part 1 (routed experts)
//                  res' = (res + norm(norm(s, g0) + norm(r, g1), g2)) * layer_scalar  -> cb_res_out (BF16)
//                  x = norm(res', g3)                                        -> out0 (BF16)
// Every FPU phase that reads a new CB format starts with compute_kernel_hw_startup (gpt-oss boundary_compute.cpp:
// the scalar reduce / broadcast need the full configuration).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_recv = get_compile_time_arg_val(0);
constexpr uint32_t cb_res = get_compile_time_arg_val(1);
constexpr uint32_t cb_g = get_compile_time_arg_val(2);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
constexpr uint32_t cb_vf = get_compile_time_arg_val(4);
constexpr uint32_t cb_a = get_compile_time_arg_val(5);  // FP32 temporaries
constexpr uint32_t cb_b = get_compile_time_arg_val(6);
constexpr uint32_t cb_ss = get_compile_time_arg_val(7);
constexpr uint32_t cb_rs = get_compile_time_arg_val(8);
constexpr uint32_t cb_res_out = get_compile_time_arg_val(9);
constexpr uint32_t cb_o0 = get_compile_time_arg_val(10);  // outputs cb_o0, cb_o0 + 1, cb_o0 + 2
constexpr uint32_t T = get_compile_time_arg_val(11);
constexpr uint32_t slots = get_compile_time_arg_val(12);
constexpr uint32_t parts = get_compile_time_arg_val(13);
constexpr uint32_t mode = get_compile_time_arg_val(14);
constexpr uint32_t inv_n = get_compile_time_arg_val(15);       // float bits of 1 / hidden
constexpr uint32_t eps = get_compile_time_arg_val(16);         // float bits
constexpr uint32_t out2_gamma = get_compile_time_arg_val(17);  // attn: out2 = n * g2 (else n)
constexpr uint32_t cb_n = get_compile_time_arg_val(18);  // attn: n (FP32, or BF16 = rounded as a ttnn RMSNorm output)
static_assert(T <= 3, "a vector plus one operand fit the 4 FP32 DST tiles");

// FAST_RECONFIG: the elementwise phases only reconfigure the unpacker / packer data formats; the scalar reduce keeps
// the full hardware startup.
#ifdef FAST_RECONFIG
#define STARTUP(a, b, o)            \
    reconfig_data_format((a), (b)); \
    pack_reconfig_data_format(o)
#else
#define STARTUP(a, b, o) compute_kernel_hw_startup((a), (b), (o))
#endif

constexpr uint32_t kScalarIters = 1;
template <int OP>
ALWI void scalar_op(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kScalarIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
ALWI void rsqrt_scalar(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, kScalarIters, DST_ACCUM_MODE, false),
        idst,
        VectorMode::R));
}

ALWI void pack_vec(uint32_t cb) {
    tile_regs_commit();
    cb_reserve_back(cb, T);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t t = 0; t < T; ++t) {
        pack_tile(t, cb);
    }
    tile_regs_release();
    cb_push_back(cb, T);
}

// cb_out <- sum over the slots of part p (slot order, identical on every device).
ALWI void sum_slots(uint32_t p, uint32_t cb_out) {
    STARTUP(cb_recv, cb_recv, cb_out);
    tile_regs_acquire();
    for (uint32_t t = 0; t < T; ++t) {
        add_init(cb_recv, cb_recv);
        add_tiles(cb_recv, cb_recv, p * T + t, (parts + p) * T + t, t);
        if constexpr (slots > 2) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv);
            for (uint32_t s = 2; s < slots; ++s) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv, (s * parts + p) * T + t, t);
            }
        }
    }
    pack_vec(cb_out);
}

// cb_ss += one tile: the elementwise sum over the T tiles of v^2.
ALWI void sumsq(uint32_t cb_v) {
    STARTUP(cb_v, cb_v, cb_ss);
    cb_wait_front(cb_v, T);
    tile_regs_acquire();
    mul_init(cb_v, cb_v);
    for (uint32_t t = 0; t < T; ++t) {
        mul_tiles(cb_v, cb_v, t, t, t);
    }
    add_binary_tile_init();
    for (uint32_t t = 1; t < T; ++t) {
        add_binary_tile(0, t, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_ss, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ss);
    pack_tile(0, cb_ss);
    tile_regs_release();
    cb_push_back(cb_ss, 1);
}

// cb_rs += rsqrt(sum(cb_ss front tile) / hidden + eps) (element 0), pops cb_ss.
ALWI void rstat() {
    cb_wait_front(cb_ss, 1);
    compute_kernel_hw_startup(cb_ss, cb_scaler, cb_rs);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_ss, cb_scaler, cb_rs);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_ss, cb_scaler, 0, 0, 0);
    reduce_uninit(cb_ss);
    binop_with_scalar_tile_init();
    scalar_op<MUL_UNARY>(0, inv_n);
    scalar_op<ADD_UNARY>(0, eps);
    rsqrt_tile_init();
    rsqrt_scalar(0);
    tile_regs_commit();
    cb_reserve_back(cb_rs, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_rs);
    pack_tile(0, cb_rs);
    tile_regs_release();
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_ss, 1);
}

// DST[0..T) = v * rs (front of cb_rs, popped) [* gamma gi]. Leaves DST acquired.
ALWI void scaled(uint32_t cb_v, int gi) {
    cb_wait_front(cb_rs, 1);
    STARTUP(cb_v, cb_rs, cb_a);
    tile_regs_acquire();
    mul_bcast_scalar_init(cb_v, cb_rs);
    for (uint32_t t = 0; t < T; ++t) {
        mul_tiles_bcast_scalar(cb_v, cb_rs, t, 0, t);
    }
    if (gi >= 0) {
        reconfig_data_format_srcb(cb_rs, cb_g);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g);
        for (uint32_t t = 0; t < T; ++t) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, gi * T + t, t);
        }
    }
    cb_pop_front(cb_rs, 1);
}

// DST[0..T) op= cb_x (reuse dest, DST -> SrcA). Call after reconfiguring SrcB to cb_x's format.
ALWI void add_vec(uint32_t cb_x) {
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x);
    for (uint32_t t = 0; t < T; ++t) {
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x, t, t);
    }
}
ALWI void mul_vec(uint32_t cb_x, uint32_t first) {
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x);
    for (uint32_t t = 0; t < T; ++t) {
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x, first + t, t);
    }
}

void kernel_main() {
#ifdef FAST_RECONFIG
    compute_kernel_hw_startup(cb_res, cb_res, cb_res_out);
#endif
    static_assert(mode == 0 || slots >= 2, "the all-reduce sums at least two slots");
    if constexpr (mode == 0) {
        // entry: x = norm(res, g0)
        cb_wait_front(cb_g, T);
        sumsq(cb_res);
        rstat();
        scaled(cb_res, 0);
        pack_vec(cb_o0);
        cb_pop_front(cb_res, T);
        return;
    }
    cb_wait_front(cb_recv, slots * parts * T);
    cb_wait_front(cb_res, T);
    if constexpr (mode == 1) {
        const uint32_t G = out2_gamma ? 3 : 2;
        cb_wait_front(cb_g, G * T);
        // a = sum of the slots; res' = res + norm(a, g0)
        sum_slots(0, cb_a);
        sumsq(cb_a);
        rstat();
        scaled(cb_a, 0);
        reconfig_data_format_srcb(cb_g, cb_res);
        add_vec(cb_res);
        pack_vec(cb_res_out);
        cb_pop_front(cb_a, T);
        cb_pop_front(cb_res, T);
        // n = norm(res', 1) -> cb_n
        sumsq(cb_res_out);
        rstat();
        scaled(cb_res_out, -1);
        pack_vec(cb_n);
        // out0 = n * vf (FP32), out1 = n * g1, out2 = n * g2 or n
        cb_wait_front(cb_n, T);
        cb_wait_front(cb_vf, T);
        STARTUP(cb_n, cb_vf, cb_o0);
        tile_regs_acquire();
        mul_init(cb_n, cb_vf);
        for (uint32_t t = 0; t < T; ++t) {
            mul_tiles(cb_n, cb_vf, t, t, t);
        }
        pack_vec(cb_o0);
        reconfig_data_format_srcb(cb_vf, cb_g);
        tile_regs_acquire();
        mul_init(cb_n, cb_g);
        for (uint32_t t = 0; t < T; ++t) {
            mul_tiles(cb_n, cb_g, t, T + t, t);
        }
        pack_vec(cb_o0 + 1);
        tile_regs_acquire();
        if constexpr (out2_gamma) {
            for (uint32_t t = 0; t < T; ++t) {
                mul_tiles(cb_n, cb_g, t, 2 * T + t, t);
            }
        } else {
            copy_init(cb_n);
            for (uint32_t t = 0; t < T; ++t) {
                copy_tile(cb_n, t, t);
            }
        }
        pack_vec(cb_o0 + 2);
        cb_pop_front(cb_n, T);
        cb_pop_front(cb_vf, T);
    } else {
        cb_wait_front(cb_g, 4 * T);
        const uint32_t layer_scalar = get_arg_val<uint32_t>(0);
        // s, r = slot sums of the shared / routed partials; c = norm(s, g0) + norm(r, g1) -> cb_a
        sum_slots(0, cb_a);
        sum_slots(1, cb_b);
        sumsq(cb_a);
        sumsq(cb_b);
        rstat();
        rstat();
        scaled(cb_a, 0);
        pack_vec(cb_a);  // norm(s, g0), behind s in cb_a
        cb_pop_front(cb_a, T);
        scaled(cb_b, 1);
        reconfig_data_format_srcb(cb_g, cb_a);
        add_vec(cb_a);
        pack_vec(cb_b);  // c, behind r in cb_b
        cb_pop_front(cb_a, T);
        cb_pop_front(cb_b, T);
        // res' = (res + norm(c, g2)) * layer_scalar
        sumsq(cb_b);
        rstat();
        scaled(cb_b, 2);
        reconfig_data_format_srcb(cb_g, cb_res);
        add_vec(cb_res);
        binop_with_scalar_tile_init();
        for (uint32_t t = 0; t < T; ++t) {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_binop_with_scalar,
                (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
                t,
                VectorMode::RC,
                layer_scalar));
        }
        pack_vec(cb_res_out);
        cb_pop_front(cb_b, T);
        cb_pop_front(cb_res, T);
        // x = norm(res', g3)
        sumsq(cb_res_out);
        rstat();
        scaled(cb_res_out, 3);
        pack_vec(cb_o0);
    }
    cb_pop_front(cb_recv, slots * parts * T);
}
