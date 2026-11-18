// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K2 compute (one core per token tile-row): MS = sum_k M_k, QS = sum_k Q_k (exact fp32 adds of
// unpacked-to-dest tiles); R = rsqrt(sum_cols(QS) / (4D) + eps); pp = (sigmoid(MS * R * SCALE + BASE) * MUL + ADD)
// (columns 0..3 pre, 4..7 post), written as ONE tile per token tile-row (K3 / K5 readers replicate the columns exactly
// on the RISC). CT: 0 cb_cst (ones), 1 cb_part, 2 cb_ms, 3 cb_qs, 4 cb_pp, 5 cb_cst2 (SCALE, BASE, MUL, ADD), 6 NCH, 7
// INV_N bits, 8 EPSB bits.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_cst = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ms = get_compile_time_arg_val(2);
    constexpr uint32_t cb_qs = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pp = get_compile_time_arg_val(4);
    constexpr uint32_t cb_cst2 = get_compile_time_arg_val(5);
    constexpr uint32_t NCH = get_compile_time_arg_val(6);
    constexpr uint32_t INV_N = get_compile_time_arg_val(7);
    constexpr uint32_t EPSB = get_compile_time_arg_val(8);
    compute_kernel_hw_startup(cb_part, cb_cst, cb_pp);
    CircularBuffer cst(cb_cst), cst2(cb_cst2), part(cb_part), ms(cb_ms), qs(cb_qs), pp(cb_pp);
    cst.wait_front(1);
    part.wait_front(2 * NCH);
    // ---- MS (dest 0), QS (dest 2)
    tile_regs_acquire();
    reconfig_data_format_srca(cb_part);
    copy_tile_to_dst_init_short(cb_part);
    copy_tile(cb_part, 0, 0);
    copy_tile(cb_part, NCH, 2);
    for (uint32_t k = 1; k < NCH; ++k) {
        copy_tile_to_dst_init_short(cb_part);
        copy_tile(cb_part, k, 1);
        copy_tile(cb_part, NCH + k, 3);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(2, 3, 2);
    }
    tile_regs_commit();
    tile_regs_wait();
    ms.reserve_back(1);
    qs.reserve_back(1);
    pack_reconfig_data_format(cb_ms);
    pack_tile(0, cb_ms);
    pack_tile(2, cb_qs);
    ms.push_back(1);
    qs.push_back(1);
    tile_regs_release();
    // ---- pp (exact fp32 SFPU chain: only the sum over the columns of Q uses the FPU)
    ms.wait_front(1);
    qs.wait_front(1);
    cst2.wait_front(4);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_qs, cb_cst);
    matmul_init(cb_qs, cb_cst);
    matmul_tiles(cb_qs, cb_cst, 0, 0, 0);  // every column = sum of the squares of the token
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_N);
    add_unary_tile(0, EPSB);
    rsqrt_tile_init();
    rsqrt_tile(0);
    reconfig_data_format_srca(cb_ms);
    copy_tile_to_dst_init_short(cb_ms);
    copy_tile(cb_ms, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);  // m * r
    reconfig_data_format_srca(cb_cst2);
    copy_tile_to_dst_init_short(cb_cst2);
    copy_tile(cb_cst2, 0, 1);  // SCALE
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    copy_tile(cb_cst2, 1, 1);  // BASE
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    copy_tile_to_dst_init_short(cb_cst2);
    copy_tile(cb_cst2, 2, 1);  // MUL
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    copy_tile(cb_cst2, 3, 1);  // ADD
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pp.reserve_back(1);
    pack_tile(0, cb_pp);
    pp.push_back(1);
    tile_regs_release();
}
