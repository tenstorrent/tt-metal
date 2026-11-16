// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerNorm reader. Pushes, in the order the compute kernel consumes them: the scaler
// tile, the eps tile, then each row's a/b tiles (blk at a time). gamma and beta (Wt
// tiles each) follow the first row's inputs, as in the stock reader.
//
// gamma/beta are row-major bf16 [1, 1, Wt, 32] (the model's LayerNorm weights): page k
// holds the 32 weights of tile column k. They go to row 0 of faces 0 and 1 of tile k;
// the row broadcast reads only that row.
//
// Compile-time args: 0 Wt, 1 blk, 2 eps (bf16 bits in the high half), 3+ accessors a, b, gamma, beta
// Runtime args: 0 a_addr, 1 b_addr, 2 gamma_addr, 3 beta_addr, 4 row_start, 5 num_rows

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

constexpr uint32_t CB_A = 0;
constexpr uint32_t CB_B = 1;
constexpr uint32_t CB_SCALER = 2;
constexpr uint32_t CB_EPS = 3;
constexpr uint32_t CB_GAMMA = 4;
constexpr uint32_t CB_BETA = 5;

// n tiles of a and b at the same pages, one barrier.
template <typename AccA, typename AccB>
FORCE_INLINE void read_ab(const Noc& noc, const AccA& a, const AccB& b, uint32_t first_page, uint32_t n) {
    constexpr uint32_t a_bytes = get_tile_size(CB_A);
    constexpr uint32_t b_bytes = get_tile_size(CB_B);
    CircularBuffer cb_a(CB_A);
    CircularBuffer cb_b(CB_B);
    cb_a.reserve_back(n);
    cb_b.reserve_back(n);
    uint32_t da = cb_a.get_write_ptr();
    uint32_t db = cb_b.get_write_ptr();
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(a, CoreLocalMem<uint32_t>(da), a_bytes, {.page_id = first_page + i}, {});
        noc.async_read(b, CoreLocalMem<uint32_t>(db), b_bytes, {.page_id = first_page + i}, {});
        da += a_bytes;
        db += b_bytes;
    }
    noc.async_read_barrier();
    cb_a.push_back(n);
    cb_b.push_back(n);
}

// Wt row-major sticks (32 bf16 = 64 B) to row 0 of faces 0 and 1 of Wt tiles.
template <uint32_t Wt, uint32_t cb_id, typename Acc>
FORCE_INLINE void read_weight_row(const Noc& noc, const Acc& acc) {
    constexpr uint32_t tile_bytes = get_tile_size(cb_id);
    constexpr uint32_t stick_bytes = 64;
    CircularBuffer cb(cb_id);
    cb.reserve_back(Wt);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t k = 0; k < Wt; ++k) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(base + k * tile_bytes), stick_bytes, {.page_id = k}, {});
    }
    noc.async_read_barrier();
    // Elements 16..31 landed at face 0 row 1; move them to face 1 row 0 (byte 512).
    for (uint32_t k = 0; k < Wt; ++k) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + k * tile_bytes);
        for (uint32_t w = 0; w < 8; ++w) {
            t[128 + w] = t[8 + w];
        }
    }
    cb.push_back(Wt);
}

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto be_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);

    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    const auto gamma = TensorAccessor(g_args, gamma_addr);
    const auto beta = TensorAccessor(be_args, beta_addr);
    Noc noc;

    // Reduce scaler: zeros, then row 0 of each face = 1.0 (bf16), as the stock reader.
    {
        CircularBuffer cb(CB_SCALER);
        cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for (uint32_t i = 0; i < 512; ++i) {
            p[i] = 0;
        }
        for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t c = 0; c < 8; ++c) {
                p[face * 128 + c] = 0x3f803f80;
            }
        }
        cb.push_back(1);
    }
    generate_bcast_col_scalar(CircularBuffer(CB_EPS), eps_bits);

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t base = (row_start + r) * Wt;
        for (uint32_t t = 0; t < Wt; t += blk) {
            read_ab(noc, a, b, base + t, blk);
        }
        if (r == 0) {
            read_weight_row<Wt, CB_GAMMA>(noc, gamma);
            read_weight_row<Wt, CB_BETA>(noc, beta);
        }
    }
}
