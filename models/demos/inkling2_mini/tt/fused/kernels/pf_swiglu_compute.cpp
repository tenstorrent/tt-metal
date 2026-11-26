// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill softcapped SwiGLU, compute: out = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25), with the rounding points of
// moe.softcapped_swiglu's two ttnn.mul calls (binary_ng's SFPU MUL on bf16: 16-bit DST, operands unpacked to DST as
// fp32): each operand's activations run in DST and are packed bf16, the SFPU multiplies, the post scale runs in
// DST, the result is packed bf16.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
// in -> DST 0 -> fn -> pack to out (one tile; the caller waits / pops in)
template <typename F>
FORCE_INLINE void act(uint32_t in, uint32_t out, F fn) {
    cb_reserve_back(out, 1);
    reconfig_data_format_srca(in);
    pack_reconfig_data_format(out);
    copy_init(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    fn();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
}
}  // namespace

// a * b on the SFPU, as binary_ng's SFPU MUL (both operands copied to DST), then `post` (the caller's DST op)
template <typename F>
FORCE_INLINE void mul(uint32_t a, uint32_t b, uint32_t out, F post) {
    cb_wait_front(a, 1);
    cb_wait_front(b, 1);
    cb_reserve_back(out, 1);
    pack_reconfig_data_format(out);
    tile_regs_acquire();
    reconfig_data_format_srca(a);
    copy_init(a);
    copy_tile(a, 0, 0);
    reconfig_data_format_srca(b);
    copy_init(b);
    copy_tile(b, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    post();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(a, 1);
    cb_pop_front(b, 1);
}

void kernel_main() {
    constexpr uint32_t HT = get_named_compile_time_arg_val("ht");
    constexpr uint32_t g_scale = get_named_compile_time_arg_val("g_scale");  // fp32 bits of 1 / GATE_SOFTCAP
    constexpr uint32_t u_scale = get_named_compile_time_arg_val("u_scale");  // fp32 bits of 1 / UP_SOFTCAP
    constexpr uint32_t post = get_named_compile_time_arg_val("post");        // fp32 bits of GATE_SOFTCAP UP_SOFTCAP
    constexpr uint32_t cb_g = 0, cb_u = 1, cb_a = 2, cb_b = 3, cb_p = 4, cb_c = 5, cb_out = 16;
    const uint32_t n_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_g, cb_u, cb_out);
    for (uint32_t t = 0; t < n_rows * HT; ++t) {
        cb_wait_front(cb_g, 1);
        act(cb_g, cb_a, [] {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, g_scale);
            tanh_tile_init();
            tanh_tile(0);
        });
        act(cb_g, cb_b, [] {
            sigmoid_tile_init();
            sigmoid_tile(0);
        });
        cb_pop_front(cb_g, 1);
        mul(cb_a, cb_b, cb_p, [] {});
        cb_wait_front(cb_u, 1);
        act(cb_u, cb_c, [] {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, u_scale);
            tanh_tile_init();
            tanh_tile(0);
        });
        cb_pop_front(cb_u, 1);
        mul(cb_p, cb_c, cb_out, [] {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, post);
        });
    }
}
