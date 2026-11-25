// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched decode dense MoE gate|up + bias + SwiGLU + routing weights in one DRAM-streaming program (experts/decode.py
// _dense_gate_up_stream), reader (BRISC). Replaces the gate and up DRAM-sharded matmuls and the dense SwiGLU op.
// Units: the NA column tiles n of the [32, E * I_local] activation, core c of NC taking [c NA / NC, (c + 1) NA / NC).
// A unit's weights are the grouped MoE's relaid gate|up page of (expert e, local column j) = the dense columns of tile
// n (gmoe_relayout.cpp: [k][gate, up], GMOE_SPLIT consecutive pages): e = the expert of the tile's first column,
// j = n - g0(e) (the tile two experts share is the even expert's last local tile). Pushes:
//   cb_x: the 32 rows' K tiles (BFP8, once);
//   cb_r: the routing weights tile [32 tokens, 32 experts] (BF16, once);
//   per unit: cb_w its 2 KT weight tiles (BFP4), cb_b the gate / up bias tiles n (32 equal rows, BF16), cb_e the
//   one-hot expand tile (1.0 at (expert of column cc, cc)); up to DEPTH units buffered, one unit's reads outstanding.
//
// common runtime args: [x, gate_up (relaid), gate_bias, up_bias, w]
// compile-time args: [cb_x, cb_w, cb_b, cb_r, cb_e, NC, GX, KT, NA, I_LOCAL, NU, accessors (x, gate_up, gate_bias,
//                     up_bias, w)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmoe_common.hpp"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gu_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t gb_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t ub_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(4);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_r = get_compile_time_arg_val(3);
    constexpr uint32_t cb_e = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr uint32_t GX = get_compile_time_arg_val(6);
    constexpr uint32_t KT = get_compile_time_arg_val(7);
    constexpr uint32_t NA = get_compile_time_arg_val(8);       // activation column tiles (E * I_local / 32)
    constexpr uint32_t I_LOCAL = get_compile_time_arg_val(9);  // columns per expert
    constexpr uint32_t NU = get_compile_time_arg_val(10);      // relaid units per expert (23)
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto gu_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gb_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto ub_args = TensorAccessorArgs<gb_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<ub_args.next_compile_time_args_offset()>();
    constexpr uint32_t x_tile = 1088;  // BFP8_b
    constexpr uint32_t w_tile = 576;   // BFP4_b
    constexpr uint32_t b_tile = 2048;  // BF16
    constexpr uint32_t SPLIT = GMOE_SPLIT;
    constexpr uint32_t PW = 2 * KT;  // weight tiles per unit
    constexpr uint32_t sp = PW * w_tile / SPLIT;
    static_assert(PW % SPLIT == 0 && sp % 64 == 0);

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t n0 = c * NA / NC, n1 = (c + 1) * NA / NC;
    const auto s_x = TensorAccessor(x_args, x_addr, x_tile);
    const auto s_gu = TensorAccessor(gu_args, gu_addr, sp);
    const auto s_gb = TensorAccessor(gb_args, gb_addr, b_tile);
    const auto s_ub = TensorAccessor(ub_args, ub_addr, b_tile);
    const auto s_w = TensorAccessor(w_args, w_addr, b_tile);
    constexpr uint8_t NOC_B = 1;

    // the rows and the routing weights (once)
    cb_reserve_back(cb_x, KT);
    const uint32_t xl1 = get_write_ptr(cb_x);
    for (uint32_t k = 0; k < KT; ++k) {
        noc_async_read(s_x.get_noc_addr(k, 0, k % 2), xl1 + k * x_tile, x_tile, k % 2);
    }
    cb_reserve_back(cb_r, 1);
    noc_async_read(s_w.get_noc_addr(0, 0, 0), get_write_ptr(cb_r), b_tile, 0);
    noc_async_read_barrier(0);
    noc_async_read_barrier(NOC_B);
    cb_push_back(cb_x, KT);
    cb_push_back(cb_r, 1);

    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t n = n0; n < n1; ++n) {
        const uint32_t e = n * 32 / I_LOCAL, u = e * NU + (n - gmoe::g0(e));
        cb_reserve_back(cb_w, PW);
        cb_reserve_back(cb_b, 2);
        cb_reserve_back(cb_e, 1);
        const uint32_t wl1 = get_write_ptr(cb_w), bl1 = get_write_ptr(cb_b), el1 = get_write_ptr(cb_e);
        for (uint32_t q = 0; q < SPLIT; ++q) {
            const uint8_t nn = q % 2 ? NOC_B : 0;
            noc_async_read(s_gu.get_noc_addr(u * SPLIT + q, 0, nn), wl1 + q * sp, sp, nn);
        }
        noc_async_read(s_gb.get_noc_addr(n, 0, 0), bl1, b_tile, 0);
        noc_async_read(s_ub.get_noc_addr(n, 0, NOC_B), bl1 + b_tile, b_tile, NOC_B);
        for (uint32_t z = 0; z < b_tile; z += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, el1 + z, MEM_ZEROS_SIZE, 0);
        }
        noc_async_read_barrier(0);
        // the expand tile: column cc of tile n belongs to expert (32 n + cc) / I_LOCAL
        volatile tt_l1_ptr uint16_t* et = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(el1);
        for (uint32_t cc = 0; cc < 32; ++cc) {
            const uint32_t ec = (n * 32 + cc) / I_LOCAL;
            et[(((ec / 16) * 2 + cc / 16) * 256) + (ec % 16) * 16 + cc % 16] = 0x3F80;
        }
        noc_async_read_barrier(NOC_B);
        cb_push_back(cb_w, PW);
        cb_push_back(cb_b, 2);
        cb_push_back(cb_e, 1);
    }
    // (both NOCs' global barriers before exit: gmoe_gu_reader.cpp)
    noc_async_read_barrier(0);
    noc_async_read_barrier(NOC_B);
}
