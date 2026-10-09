// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// N-core reader. Pushes the constant tiles the stock RMSNorm reader
// (reader_unary_interleaved_ln_rm_gb.cpp) builds, then this core's q tile rows (bf8, read
// straight from the full q buffer, so no slice/concat).
//
//   scaler: zeros, then 1.0 (bf16) in row 0 of each face
//           (calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>, reduce_helpers_dataflow.inl)
//   eps:    generate_bcast_col_scalar(eps_fp32_bits): bf16 = high half of the fp32 bits (truncation)
//   gamma:  row-major [1, 1, 4, 32] bf16; stick k (64 B) -> tile k at byte 0, then bytes 32..63
//           copied to face 1 row 0 (byte 512), as the stock reader does
//
// Compile-time args: 0 eps fp32 bits, 1+ accessors src, gamma
// Runtime args: 0 src_addr, 1 gamma_addr, 2 num_rows, 3 h0, 4 j0, 5 tl0, 6 sl_t, 7 s0_t, 8 s1_t, 9 ht,
//               10 n_heads, 11 src_base, 12 src_row_stride, 13 src_head_stride, 14 src_local
// The source tile of (head h, tile row t, column i) is page src_base + t * src_row_stride + h * src_head_stride + i:
// the q buffer itself ([1, H, S, 128]: 0, 4, S/32 * 4) or the fused QKV buffer ([1, 1, S, W]: column offset, W/32, 4).
// src_local = 1: the source holds only this column's two slices ([1, 1, 2 * S/(2*cols), W], local slice j first), so
// its tile row is j * sl_t + tl instead of the natural t(); cos/sin and the destination keep the natural rows.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "qprep_common.hpp"

using namespace qprep;

void kernel_main() {
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto g_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    RowWalker w{
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10)};

    const uint32_t src_base = get_arg_val<uint32_t>(11);
    const uint32_t src_row_stride = get_arg_val<uint32_t>(12);
    const uint32_t src_head_stride = get_arg_val<uint32_t>(13);
    const uint32_t src_local = get_arg_val<uint32_t>(14);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto gamma = TensorAccessor(g_args, gamma_addr);
    Noc noc;

    // Reduce scaler.
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

    // Gamma: one row, resident for the whole kernel.
    {
        constexpr uint32_t tile_bytes = get_tile_size(CB_GAMMA);
        constexpr uint32_t stick_bytes = 64;
        CircularBuffer cb(CB_GAMMA);
        cb.reserve_back(WT);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t k = 0; k < WT; ++k) {
            noc.async_read(gamma, CoreLocalMem<uint32_t>(base + k * tile_bytes), stick_bytes, {.page_id = k}, {});
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < WT; ++k) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + k * tile_bytes);
            for (uint32_t i = 0; i < 8; ++i) {
                t[128 + i] = t[8 + i];
            }
        }
        cb.push_back(WT);
    }

    constexpr uint32_t q_bytes = get_tile_size(CB_QIN);
    CircularBuffer cb_q(CB_QIN);
    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t src_t = src_local ? (w.j * w.sl_t + w.tl) : w.t();
        const uint32_t page = src_base + src_t * src_row_stride + w.h * src_head_stride;
        cb_q.reserve_back(WT);
        uint32_t dst = cb_q.get_write_ptr();
        for (uint32_t i = 0; i < WT; ++i) {
            noc.async_read(q, CoreLocalMem<uint32_t>(dst), q_bytes, {.page_id = page + i}, {});
            dst += q_bytes;
        }
        noc.async_read_barrier();
        cb_q.push_back(WT);
        w.next();
    }
}
