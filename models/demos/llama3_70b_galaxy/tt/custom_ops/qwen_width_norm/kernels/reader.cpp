// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// qwen_width_norm reader: constant tiles (as qwen_q_prep norm_reader / the stock RMSNorm reader build them), then
// this core's x chunk row by row, and for each row the gathered partials once all gx have landed in CB_GATH.
//
//   scaler: zeros, then 1.0 (bf16) in row 0 of each face
//   eps:    generate_bcast_col_scalar(eps_fp32_bits) (bf16 = high half of the fp32 bits)
//   gamma:  row-major [1, 1, Wt, 32] bf16; stick c0 + k (64 B) -> tile k at byte 0, bytes 32..63 copied to face 1
//           row 0 (byte 512)
//   x:      tile row (row0 + r), tiles c0 .. c0 + w - 1 (page (row0 + r) * Wt + c0 + i)
//
// CB_GATH holds two rows of gx fp32 tiles; peers write row r's partials into the half r % 2 directly (the writer
// kernel of every core of the row group). This kernel only does the bookkeeping: once SEM_RECV >= gx * (r + 1) it
// reserves and pushes gx tiles, which are the half the peers wrote (reserve/push alternate the halves in the same
// order on every core).
//
// Compile-time args: wnorm_common.hpp prefix, then accessors x, gamma
// Common runtime args: 0 x_addr, 1 gamma_addr (no per-core args: the work comes from the logical coords)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "wnorm_common.hpp"

using namespace wnorm;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t gx = get_compile_time_arg_val(1);
    constexpr uint32_t gy = get_compile_time_arg_val(2);
    constexpr uint32_t Rt = get_compile_time_arg_val(3);
    constexpr uint32_t maxw = get_compile_time_arg_val(4);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<CT_NEXT>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_common_arg_val<uint32_t>(1);
    const Work wk = core_work(get_absolute_logical_x(), get_absolute_logical_y(), Wt, gx, Rt, gy);
    const uint32_t row0 = wk.row0, num_rows = wk.n, c0 = wk.c0, w = wk.w;
    const auto x = TensorAccessor(x_args, x_addr);
    const auto gamma = TensorAccessor(g_args, gamma_addr);
    Noc noc;

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

    {
        constexpr uint32_t tile_bytes = get_tile_size(CB_GAMMA);
        constexpr uint32_t stick_bytes = 64;
        CircularBuffer cb(CB_GAMMA);
        cb.reserve_back(w);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t k = 0; k < w; ++k) {
            noc.async_read(gamma, CoreLocalMem<uint32_t>(base + k * tile_bytes), stick_bytes, {.page_id = c0 + k}, {});
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < w; ++k) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + k * tile_bytes);
            for (uint32_t i = 0; i < 8; ++i) {
                t[128 + i] = t[8 + i];
            }
        }
        cb.push_back(w);
    }

    constexpr uint32_t x_bytes = get_tile_size(CB_X);
    CircularBuffer cb_x(CB_X);
    CircularBuffer cb_g(CB_GATH);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_RECV));

    auto push_row = [&](uint32_t r) {
        const uint32_t page = (row0 + r) * Wt + c0;
        cb_x.reserve_back(maxw);  // a row takes maxw slots (w used) so rows never wrap the CB end
        uint32_t dst = cb_x.get_write_ptr();
        for (uint32_t i = 0; i < w; ++i) {
            noc.async_read(x, CoreLocalMem<uint32_t>(dst), x_bytes, {.page_id = page + i}, {});
            dst += x_bytes;
        }
        noc.async_read_barrier();
        cb_x.push_back(maxw);
    };

    // CB_X holds two rows: rows 0 and 1 up front, row r + 2 after row r's partials are handed to compute.
    for (uint32_t r = 0; r < num_rows && r < 2; ++r) {
        push_row(r);
    }
    for (uint32_t r = 0; r < num_rows; ++r) {
        noc_semaphore_wait_min(recv, gx * (r + 1));
        cb_g.reserve_back(gx);
        cb_g.push_back(gx);
        if (r + 2 < num_rows) {
            push_row(r + 2);
        }
    }
}
