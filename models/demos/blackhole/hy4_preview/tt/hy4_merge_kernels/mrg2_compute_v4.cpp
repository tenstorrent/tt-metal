// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode cross-column merge, compute for the fused exchange+merge program (fuse-sparse-merge A2): same maths as
// mrg_compute.cpp, but the partials come from the landing buffer CB_A = [8 source slots][TPC o tiles, ms tile, l tile]
// (written by the fabric / the own-slot read) and CB_ST holds only the sink tile.
//   w_c = l_c exp(ms_c - sink); den = sum_c w_c + 1 (MAC of ones x w_c); out_j = (sum_c o_{c,j} w_c) * rden.
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/experimental/eltwise_mul_scalar.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

constexpr uint32_t TPC = get_compile_time_arg_val(0);
constexpr uint32_t CB_ST = get_compile_time_arg_val(1);
constexpr uint32_t CB_A = get_compile_time_arg_val(2);
constexpr uint32_t CB_W = get_compile_time_arg_val(3);
constexpr uint32_t CB_R = get_compile_time_arg_val(4);
constexpr uint32_t CB_ONE = get_compile_time_arg_val(5);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
constexpr uint32_t SLOTSEM = get_compile_time_arg_val(7);
constexpr uint32_t FULL =
    get_compile_time_arg_val(8);  // 1: SFPU ops over all four faces (compact layout with > 16 used rows)
constexpr VectorMode VM = FULL ? VectorMode::RC : VectorMode::R;
constexpr uint32_t NC = 8;
constexpr uint32_t NB = 4;
constexpr uint32_t LT = TPC + 2;  // tiles per source slot

void kernel_main() {
    CircularBuffer st(CB_ST), a(CB_A), w(CB_W), r(CB_R), one(CB_ONE), o(CB_OUT);
    compute_kernel_hw_startup(CB_A, CB_ONE, CB_OUT);
    st.wait_front(1);
    if constexpr (!SLOTSEM) {
        a.wait_front(NC * LT);
    }
    w.reserve_back(NC);
    {
        DeviceZoneScopedN("XM_A");
        // B26: slot c is processed as soon as the reader pushed it (SLOTSEM): w_c = l_c exp(ms_c - sink), one tile per
        // DST window (same per-tile ops as the batched version: bit-identical)
        for (uint32_t c = 0; c < NC; ++c) {
            if constexpr (SLOTSEM) {
                a.wait_front((c + 1) * LT);
            }
            reconfig_data_format(CB_A, CB_ST);
            sub_init(CB_A, CB_ST);
            tile_regs_acquire();
            sub_tiles(CB_A, CB_ST, c * LT + TPC, 0, 0);
            exp_tile_init();
            exp_tile<false, false, InputClamping::ClampToNegative, 8>(0, VM);
            deepseek_binary_dest_reuse_tiles_init(CB_A);
            deepseek_binary_dest_reuse_tiles(CB_A, c * LT + TPC + 1, 0);
            tile_regs_commit();
            pack_reconfig_data_format(CB_W);
            tile_regs_wait();
            pack_tile<true>(0, CB_W, c);
            tile_regs_release();
        }
    }
    w.push_back(NC);
    w.wait_front(NC);
    one.wait_front(1);
    r.reserve_back(1);
    {
        DeviceZoneScopedN("XM_D");
        reconfig_data_format(CB_ONE, CB_W);
        bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(CB_ONE, CB_W);
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
            CB_ONE, CB_W, 1)));
        tile_regs_acquire();
        for (uint32_t c = 0; c < NC; ++c) {
            mul_tiles_bcast_cols(CB_ONE, CB_W, 0, c, 0);
        }
        binop_with_scalar_tile_init();
        add_unary_tile(0, 0x3f800000u);  // + 1.0 (the sink term e^0)
        recip_tile_init();
        recip_tile(0, VM);
        tile_regs_commit();
        pack_reconfig_data_format(CB_R);
        tile_regs_wait();
        pack_tile(0, CB_R);
        tile_regs_release();
        r.push_back(1);
        r.wait_front(1);
    }
    if constexpr (SLOTSEM) {
        a.wait_front(NC * LT);
    }
    DeviceZoneScopedN("XM_J");
    for (uint32_t j = 0; j < TPC; ++j) {
        reconfig_data_format(CB_A, CB_W);
        bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(CB_A, CB_W);
        MATH((
            llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(CB_A, CB_W, 1)));
        tile_regs_acquire();
        for (uint32_t c = 0; c < NC; ++c) {
            mul_tiles_bcast_cols(CB_A, CB_W, c * LT + j, c, 0);
        }
        deepseek_binary_dest_reuse_tiles_init(CB_R);
        deepseek_binary_dest_reuse_tiles(CB_R, 0, 0);
        tile_regs_commit();
        o.reserve_back(1);
        pack_reconfig_data_format(CB_OUT);
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        o.push_back(1);
    }
    a.pop_front(NC * LT);
    st.pop_front(1);
    w.pop_front(NC);
    r.pop_front(1);
}
