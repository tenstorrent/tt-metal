// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pick_combine reader (custom_ops/pick_combine/op.py): per job (block b of 32 tokens, slice s of sw x 32 columns),
// pick by pick, the block's 32 rows of the slice out of the expert output table (ROW_MAJOR [max_buf, H] bf16; row
// pos[t*k + j], columns [s*S, (s+1)*S): one contiguous sw x 64 B read per row) into cb_g_rm as a [32, S] row-major
// block the compute tilizes; a zero-weight pick's row is zero-filled instead (its table row is garbage). On the
// first job of a block the block's pick rows and weights are read (one 64-byte-aligned row each) and the k diagonal
// weight tiles diag(w[:, j]) are built into cb_w (bf16, exact), which the compute keeps until the block's last job.
#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"

namespace rct {
enum : uint32_t { CB_G_RM = 0, CB_W, CB_POS, CB_WSLAB, K, SW, SLICES, Y_ROW_BYTES, POS_ROW_BYTES, W_ROW_BYTES, END };
}

void kernel_main() {
    constexpr uint32_t cb_g_rm = get_compile_time_arg_val(rct::CB_G_RM);
    constexpr uint32_t cb_w = get_compile_time_arg_val(rct::CB_W);
    constexpr uint32_t cb_pos = get_compile_time_arg_val(rct::CB_POS);
    constexpr uint32_t cb_wslab = get_compile_time_arg_val(rct::CB_WSLAB);
    constexpr uint32_t k = get_compile_time_arg_val(rct::K);
    constexpr uint32_t sw = get_compile_time_arg_val(rct::SW);
    constexpr uint32_t slices = get_compile_time_arg_val(rct::SLICES);
    constexpr uint32_t y_row_bytes = get_compile_time_arg_val(rct::Y_ROW_BYTES);
    constexpr uint32_t pos_row_bytes = get_compile_time_arg_val(rct::POS_ROW_BYTES);
    constexpr uint32_t w_row_bytes = get_compile_time_arg_val(rct::W_ROW_BYTES);
    constexpr auto y_args = TensorAccessorArgs<rct::END>();
    constexpr auto pos_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();

    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t job0 = get_arg_val<uint32_t>(3);
    const uint32_t jobs = get_arg_val<uint32_t>(4);

    constexpr uint32_t S_bytes = sw * tt::constants::TILE_WIDTH * 2;  // a row's slice: sw x 64 B
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_elems = tt::constants::FACE_HW;  // 256
    const auto y = TensorAccessor(y_args, y_addr, y_row_bytes);
    const auto pos = TensorAccessor(pos_args, pos_addr, pos_row_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_row_bytes);

    cb_reserve_back(cb_pos, 1);
    cb_reserve_back(cb_wslab, 1);
    const uint32_t pos_l1 = get_write_ptr(cb_pos);
    const uint32_t ws_l1 = get_write_ptr(cb_wslab);
    volatile tt_l1_ptr uint32_t* pos_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    volatile tt_l1_ptr uint16_t* w_p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ws_l1);

    uint32_t cur_block = 0xFFFFFFFFu;
    for (uint32_t job = job0; job < job0 + jobs; ++job) {
        const uint32_t b = job / slices;
        const uint32_t s = job % slices;
        if (b != cur_block) {
            cur_block = b;
            noc_async_read(pos.get_noc_addr(b), pos_l1, pos_row_bytes);
            noc_async_read(w.get_noc_addr(b), ws_l1, w_row_bytes);
            noc_async_read_barrier();
            // the k diagonal weight tiles diag(w[:, j]): element (t, t) sits in face (t / 16) * 3 at (t % 16) * 17
            cb_reserve_back(cb_w, k);
            volatile tt_l1_ptr uint32_t* wz = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w));
            for (uint32_t i = 0; i < k * tile_bytes / 4; ++i) {
                wz[i] = 0;
            }
            volatile tt_l1_ptr uint16_t* wt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_w));
            for (uint32_t j = 0; j < k; ++j) {
                for (uint32_t t = 0; t < tt::constants::TILE_HEIGHT; ++t) {
                    const uint32_t f = (t / tt::constants::FACE_HEIGHT) * 3u;
                    wt[j * (tile_bytes / 2) + f * face_elems + (t % tt::constants::FACE_HEIGHT) * 17u] = w_p[t * k + j];
                }
            }
            cb_push_back(cb_w, k);
        }
        const uint32_t col_off = s * S_bytes;
        for (uint32_t j = 0; j < k; ++j) {
            cb_reserve_back(cb_g_rm, sw);
            const uint32_t dst = get_write_ptr(cb_g_rm);
            for (uint32_t t = 0; t < tt::constants::TILE_HEIGHT; ++t) {
                const uint32_t row_l1 = dst + t * S_bytes;
                if (w_p[t * k + j] == 0) {  // a non-local pick (weight 0): its table row is garbage, zero it
                    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
                    for (uint32_t i = 0; i < S_bytes / 4; ++i) {
                        z[i] = 0;
                    }
                    continue;
                }
                noc_async_read(y.get_noc_addr(pos_p[t * k + j]) + col_off, row_l1, S_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_g_rm, sw);
        }
    }
}
