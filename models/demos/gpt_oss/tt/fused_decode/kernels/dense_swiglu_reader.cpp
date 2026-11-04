// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched decode dense MoE (experts/decode.py decode_forward_dense): the gate / up biases, SwiGLU and routing weights
// of the [32, E * I_local] gate / up outputs in one program, on their width shards (dense_swiglu_compute.cpp), reader
// (BRISC). The gate / up shards and the act output are CBs on the tensors' own L1 (pushed here without a copy); the
// one-hot expand tile of each column tile is built here; the routing weights' tile is read once; the
// biases ([32, E * I_local], 32 equal rows, DRAM interleaved) only their row 0: the first two face rows of faces 0 and
// 1 (64-byte reads), added by row broadcast.
//
// common runtime args: [w, gate_bias, up_bias, core column-tile offset table: per grid core (y GX + x) its first column
// tile, 4 per word as bytes / 8 (NT is a multiple of 8)]
// compile-time args: [cb_gate, cb_up, cb_e, cb_w, cb_bg, cb_ub, NT (column tiles per core), GX, I_LOCAL, accessors (w,
// gate_bias, up_bias)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gb_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ub_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_e = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ub = get_compile_time_arg_val(5);
    constexpr uint32_t NT = get_compile_time_arg_val(6);
    constexpr uint32_t GX = get_compile_time_arg_val(7);
    constexpr uint32_t I_LOCAL = get_compile_time_arg_val(8);  // columns per expert
    constexpr auto w_args = TensorAccessorArgs<9>();
    constexpr auto gb_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto ub_args = TensorAccessorArgs<gb_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t c0 = ((get_common_arg_val<uint32_t>(3 + c / 4) >> (8 * (c % 4))) & 0xFF) * 8;  // first column tile

    // the tensor-backed CBs: their data is in place
    cb_reserve_back(cb_gate, NT);
    cb_push_back(cb_gate, NT);
    cb_reserve_back(cb_up, NT);
    cb_push_back(cb_up, NT);

    const auto s_w = TensorAccessor(w_args, w_addr, tile);
    const auto s_gb = TensorAccessor(gb_args, gb_addr, tile);
    const auto s_ub = TensorAccessor(ub_args, ub_addr, tile);
    cb_reserve_back(cb_w, 1);
    noc_async_read(s_w.get_noc_addr(0), get_write_ptr(cb_w), tile);
    noc_async_read_barrier();
    cb_push_back(cb_w, 1);
    // per column tile (2-page CBs: L1 is tight on these cores): the bias rows, and the expand tile built here (zeros,
    // 1.0 at (expert of column cc, cc): column c of the [*, E * I_local] layout belongs to expert c / I_LOCAL)
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t j = 0; j < NT; ++j) {
        cb_reserve_back(cb_bg, 1);
        cb_reserve_back(cb_ub, 1);
        cb_reserve_back(cb_e, 1);
        const uint32_t bg = get_write_ptr(cb_bg), ub = get_write_ptr(cb_ub), e_l1 = get_write_ptr(cb_e);
        for (uint32_t f = 0; f < 2; ++f) {  // faces 0 and 1, rows 0 .. 1 (64 bytes)
            noc_async_read(s_gb.get_noc_addr(c0 + j) + f * 512, bg + f * 512, 64);
            noc_async_read(s_ub.get_noc_addr(c0 + j) + f * 512, ub + f * 512, 64);
        }
        for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, e_l1 + z, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* et = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(e_l1);
        for (uint32_t cc = 0; cc < 32; ++cc) {
            const uint32_t e = ((c0 + j) * 32 + cc) / I_LOCAL;
            et[(((e / 16) * 2 + cc / 16) * 256) + (e % 16) * 16 + cc % 16] = 0x3F80;
        }
        cb_push_back(cb_bg, 1);
        cb_push_back(cb_ub, 1);
        cb_push_back(cb_e, 1);
    }
}
