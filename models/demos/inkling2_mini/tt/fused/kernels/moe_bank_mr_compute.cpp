// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Multi-row bank expert step, compute (moe_bank_mr_reader.cpp has the plan layout). Every chain is the 1-row step's
// chain (moe_bank_compute.cpp): same in0 / in1 tiles in the same k order into Dest 0, so each row's outputs are
// bit-equal to running that row alone.
//   A (F_GU): for each union expert u (its column resident in cb_w), for each row r that routed u: x row r x the
//     column -> one 1x32 tile into cb_ga (order: u, then r).
//   B (F_GATE): per row: the SwiGLU chain on that row's page of cb_g32 / cb_u16 / cb_ws16 (8 experts in its own
//     order, gathered by the writer) -> cb_h16.
//   C (F_DN): per row: wait cb_tok; the 288-tile down chain over row r's hidden (cb_hid r * 288 ..) with the down
//     tiles of its expert e at slot(r, e) * 36 of cb_wd -> cb_o; one page into cb_done (the reader may refill).
// Runtime arg: flags.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"

constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_wd = get_named_compile_time_arg_val("cb_wd");
constexpr uint32_t cb_ga = get_named_compile_time_arg_val("cb_ga");
constexpr uint32_t cb_g32 = get_named_compile_time_arg_val("cb_g32");
constexpr uint32_t cb_u16 = get_named_compile_time_arg_val("cb_u16");
constexpr uint32_t cb_ws16 = get_named_compile_time_arg_val("cb_ws16");
constexpr uint32_t cb_h16 = get_named_compile_time_arg_val("cb_h16");
constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t cb_o = get_named_compile_time_arg_val("cb_o");
constexpr uint32_t cb_plan = get_named_compile_time_arg_val("cb_plan");
constexpr uint32_t cb_tok = get_named_compile_time_arg_val("cb_tok");
constexpr uint32_t cb_done = get_named_compile_time_arg_val("cb_done");
constexpr uint32_t sbk = get_named_compile_time_arg_val("sbk");
constexpr uint32_t kt_gu = get_named_compile_time_arg_val("kt_gu");
constexpr uint32_t kt_dn = get_named_compile_time_arg_val("kt_dn");
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t hid_tiles = get_named_compile_time_arg_val("hid_tiles");  // per row, 288
constexpr uint32_t F_GU = 1, F_GATE = 2, F_DN = 8;
constexpr uint32_t P_POS = 1, P_MASK = 33, P_SLOT = 65;

constexpr auto LAYOUT = CustomMmDestLayout::DefaultSplitAcc;

#ifdef STD_MM_GU
constexpr bool STD_GU = true;
#else
constexpr bool STD_GU = false;
#endif
#ifdef STD_MM_DN
constexpr bool STD_DN = true;
#else
constexpr bool STD_DN = false;
#endif

// k block [k, k + sbk) of a chain into Dest 0: in0 tiles from in0_at, in1 tiles from in1_at (both resident)
template <bool std_mm>
FORCE_INLINE void mm_block(uint32_t cb_in0, uint32_t in0_at, uint32_t cb_in1, uint32_t in1_at, bool last) {
    if constexpr (std_mm) {
        for (uint32_t kk = 0; kk < sbk; ++kk) {
            matmul_block(cb_in0, cb_in1, in0_at + kk, in1_at + kk, 0, 0, 1, 1, 1);
        }
    } else if (last) {
        custom_mm_block<LAYOUT, true>(cb_in0, cb_in1, in0_at, in1_at, 0, sbk);
    } else {
        custom_mm_block<LAYOUT, false>(cb_in0, cb_in1, in0_at, in1_at, 0, sbk);
    }
}

FORCE_INLINE void pack_out(uint32_t cb_out) {
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

void kernel_main() {
    const uint32_t flags = get_arg_val<uint32_t>(0);
    cb_wait_front(cb_plan, 1);
    const uint32_t U = read_tile_value(cb_plan, 0, 0);

    if (flags & F_GU) {
        if constexpr (STD_GU) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_ga);
            matmul_block_init(cb_x, cb_w, 0, 1, 1, 1);
        } else {
            custom_mm_block_init<false, LAYOUT>(cb_x, cb_w, cb_ga);
        }
        cb_wait_front(cb_x, rows * kt_gu);
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t mask = read_tile_value(cb_plan, 0, P_MASK + u);
            cb_wait_front(cb_w, kt_gu);
            for (uint32_t r = 0; r < rows; ++r) {
                if (!(mask & (1u << r))) {
                    continue;
                }
                tile_regs_acquire();
                for (uint32_t k = 0; k < kt_gu; k += sbk) {
                    mm_block<STD_GU>(cb_x, r * kt_gu + k, cb_w, k, k + sbk >= kt_gu);
                }
                pack_out(cb_ga);
            }
            cb_pop_front(cb_w, kt_gu);
        }
        if constexpr (!STD_GU) {
            custom_mm_block_uninit<LAYOUT>(cb_ga);
        }
        cb_pop_front(cb_x, rows * kt_gu);
    } else if (flags & F_DN) {  // down-only core: the hardware init A would have done
        if constexpr (STD_DN) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(cb_hid, cb_wd, cb_o);
        } else {
            custom_mm_block_init<false, LAYOUT>(cb_hid, cb_wd, cb_o);
            custom_mm_block_uninit<LAYOUT>(cb_o);
        }
    }

    if (flags & F_GATE) {  // B per row: same chain as moe_bank_compute.cpp on that row's page
        reconfig_data_format_srca(cb_g32);
        pack_reconfig_data_format(cb_h16);
        pack_init(cb_h16);
        for (uint32_t r = 0; r < rows; ++r) {
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
        }
    }
    if (!(flags & F_DN)) {
        return;
    }
    reconfig_data_format_srca(cb_wd);
    pack_reconfig_data_format(cb_o);
    pack_init(cb_o);
    if constexpr (STD_DN) {
        matmul_block_init(cb_hid, cb_wd, 0, 1, 1, 1);
    } else {
        custom_mm_block_init_short<false, LAYOUT>(cb_hid, cb_wd, cb_o);
    }
    cb_wait_front(cb_hid, rows * hid_tiles);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_wait_front(cb_tok, 1);
        cb_pop_front(cb_tok, 1);
        if (r == 0) {
            cb_wait_front(cb_wd, topk * kt_dn);
        }
        tile_regs_acquire();
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t slot = read_tile_value(cb_plan, 0, P_SLOT + r * topk + e);
            for (uint32_t k = 0; k < kt_dn; k += sbk) {
                mm_block<STD_DN>(
                    cb_hid, r * hid_tiles + e * kt_dn + k, cb_wd, slot * kt_dn + k, e == topk - 1 && k + sbk >= kt_dn);
            }
        }
        pack_out(cb_o);
        if (r + 1 < rows) {
            cb_reserve_back(cb_done, 1);
            cb_push_back(cb_done, 1);
        }
    }
    if constexpr (!STD_DN) {
        custom_mm_block_uninit<LAYOUT>(cb_o);
    }
    cb_pop_front(cb_hid, rows * hid_tiles);
    cb_pop_front(cb_wd, topk * kt_dn);
}
