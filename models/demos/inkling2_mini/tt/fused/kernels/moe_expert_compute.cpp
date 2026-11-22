// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE expert step, compute, one of the 72 worker cores (Blackhole custom_mm: in0 1x32 tiles, LoFi,
// fp32 dest).
//   A (all):   for each of the 8 experts, one output tile of x @ W_gu[:, col_gu] (gate col j on gate cores, up col j
//              on up cores), packed bf16 into cb_g (8 tiny tiles).
//   B (gate):  h = tanh(g/4) * sigmoid(g) * tanh(u/25) * 100 * w_e, on face 0 of a 32x32 view (8 tiny tiles
//              = 512 B), packed bf16 into cb_h16 (its first 512 B are the slot the writer sends).
//   C (all):   out[col_dn] = sum over (kt, e) of hid[kt][e] * W_dn[e][kt][col_dn], one 288-tile K chain in dest.
// Runtime args: is_gate.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_g = get_named_compile_time_arg_val("cb_g");
constexpr uint32_t cb_g32 = get_named_compile_time_arg_val("cb_g32");
constexpr uint32_t cb_u16 = get_named_compile_time_arg_val("cb_u16");
constexpr uint32_t cb_ws16 = get_named_compile_time_arg_val("cb_ws16");
constexpr uint32_t cb_h16 = get_named_compile_time_arg_val("cb_h16");
constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t cb_o = get_named_compile_time_arg_val("cb_o");
constexpr uint32_t sbk = get_named_compile_time_arg_val("sbk");
constexpr uint32_t kt_gu = get_named_compile_time_arg_val("kt_gu");
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");
constexpr uint32_t hid_tiles = get_named_compile_time_arg_val("hid_tiles");

constexpr auto LAYOUT = CustomMmDestLayout::DefaultSplitAcc;

// One output tile: in0 tiles [in0_first, in0_first + k_tiles) of cb_in0 times k_tiles streamed weight tiles.
template <uint32_t k_tiles>
FORCE_INLINE void mm_tile(uint32_t cb_in0, uint32_t in0_first, uint32_t cb_out) {
    tile_regs_acquire();
    for (uint32_t k = 0; k < k_tiles; k += sbk) {
        cb_wait_front(cb_w, sbk);
        if (k + sbk < k_tiles) {
            custom_mm_block<LAYOUT, false>(cb_in0, cb_w, in0_first + k, 0, 0, sbk);
        } else {
            custom_mm_block<LAYOUT, true>(cb_in0, cb_w, in0_first + k, 0, 0, sbk);
        }
        cb_pop_front(cb_w, sbk);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

void kernel_main() {
    const uint32_t is_gate = get_arg_val<uint32_t>(0);

    // ---- A: gate_up column for each expert
    custom_mm_block_init<false, LAYOUT>(cb_x, cb_w, cb_g);
    cb_wait_front(cb_x, kt_gu);
    for (uint32_t e = 0; e < topk; ++e) {
        mm_tile<kt_gu>(cb_x, 0, cb_g);
    }
    custom_mm_block_uninit<LAYOUT>(cb_g);
    cb_pop_front(cb_x, kt_gu);

    // ---- B: softcapped SwiGLU x routing weight, 8 experts in one tile
    if (is_gate) {
        // The 8 tiny g tiles (512 B) are face 0 of one standard 32x32 bf16 tile in cb_g32's view; u, w and h use
        // the same view, so the eltwise chain runs on the best-trodden 32x32 copy/SFPU/pack path.
        cb_reserve_back(cb_g32, 1);
        cb_push_back(cb_g32, 1);
        reconfig_data_format_srca(cb_g32);
        pack_reconfig_data_format(cb_h16);
        pack_init(cb_h16);
        cb_wait_front(cb_g32, 1);
        cb_wait_front(cb_u16, 1);
        cb_wait_front(cb_ws16, 1);
        tile_regs_acquire();
        copy_init(cb_g32);
        copy_tile(cb_g32, 0, 0);
        copy_tile(cb_g32, 0, 1);
        copy_init(cb_u16);
        copy_tile(cb_u16, 0, 2);
        copy_init(cb_ws16);
        copy_tile(cb_ws16, 0, 3);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x3e800000u);  // g / 4
        mul_unary_tile(2, 0x3d23d70au);  // u / 25
        tanh_tile_init();
        tanh_tile(0);
        tanh_tile(2);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x42c80000u);  // * 4 * 25
        mul_binary_tile_init();
        mul_binary_tile(0, 3, 0);  // * routing weight
        tile_regs_commit();
        cb_reserve_back(cb_h16, 1);
        tile_regs_wait();
        pack_tile(0, cb_h16);
        tile_regs_release();
        cb_push_back(cb_h16, 1);
        cb_pop_front(cb_g32, 1);
        cb_pop_front(cb_u16, 1);
        cb_pop_front(cb_ws16, 1);
        reconfig_data_format_srca(cb_w);
    }
    pack_reconfig_data_format(cb_o);
    pack_init(cb_o);

    // ---- C: down column over the gathered hidden of all 8 experts
    custom_mm_block_init_short<false, LAYOUT>(cb_hid, cb_w, cb_o);
    cb_wait_front(cb_hid, hid_tiles);
    mm_tile<hid_tiles>(cb_hid, 0, cb_o);
    custom_mm_block_uninit<LAYOUT>(cb_o);
    cb_pop_front(cb_hid, hid_tiles);
}
