// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for conv_silu (GDN milestone 1), one core per value head.
//
// Moves only what this head consumes: its query head's kd_t q tiles, the matching kd_t k tiles, and
// its own v_t v tiles -- for the K state rows, the K diagonal-tap rows, and the new input row.
//
// The last channel group of the output (the shifted ring) needs no separate read: row j of the new
// ring is state row j+1 for j < K-1 and the new input row at j == K-1, and both are already in L1.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_qkv = 0, cb_state = 1, cb_diag = 2, cb_work = 3, cb_shift = 4;

void kernel_main() {
    constexpr uint32_t kd_t = get_compile_time_arg_val(0);
    constexpr uint32_t v_t = get_compile_time_arg_val(1);
    constexpr uint32_t c_t = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t per_core = get_compile_time_arg_val(4);

    constexpr auto qkv_a = TensorAccessorArgs<5>();
    constexpr auto state_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto diag_a = TensorAccessorArgs<state_a.next_compile_time_args_offset()>();

    // Runtime args, matching conv_silu.py's rt(): 0 qkv, 1 state, 2 diag, 3 nv, 4 gqa, 9 head.
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t state_addr = get_arg_val<uint32_t>(1);
    const uint32_t diag_addr = get_arg_val<uint32_t>(2);
    const uint32_t nv = get_arg_val<uint32_t>(3);
    const uint32_t gqa = get_arg_val<uint32_t>(4);
    const uint32_t hv = get_arg_val<uint32_t>(9);

    const uint32_t qk_h = hv / gqa;
    const uint32_t nk = nv / gqa;

    const uint32_t tb = get_tile_size(cb_qkv);
    const auto qkv_acc = TensorAccessor(qkv_a, qkv_addr, tb);
    const auto st_acc = TensorAccessor(state_a, state_addr, tb);
    const auto di_acc = TensorAccessor(diag_a, diag_addr, tb);

    // This head's channel columns within one row of c_t tiles: q block, k block, then the v block.
    const uint32_t q_col = qk_h * kd_t;
    const uint32_t k_col = nk * kd_t + qk_h * kd_t;
    const uint32_t v_col = 2 * nk * kd_t + hv * v_t;

    Noc noc;

    // One destination page offset per (kind, channel): q, then k, then v, contiguously.
    auto read_slice = [&](const auto& acc, CircularBuffer& cb, uint32_t row, uint32_t dst0) {
        uint32_t d = dst0;
        for (uint32_t i = 0; i < kd_t; i++) {
            noc.async_read(acc, cb, tb, {.page_id = row * c_t + q_col + i}, {.offset_bytes = d++ * tb});
        }
        for (uint32_t i = 0; i < kd_t; i++) {
            noc.async_read(acc, cb, tb, {.page_id = row * c_t + k_col + i}, {.offset_bytes = d++ * tb});
        }
        for (uint32_t i = 0; i < v_t; i++) {
            noc.async_read(acc, cb, tb, {.page_id = row * c_t + v_col + i}, {.offset_bytes = d++ * tb});
        }
    };

    // ---- the new input row: a [1,1,C] tensor, so row 0 ---------------------
    {
        CircularBuffer cb(cb_qkv);
        cb.reserve_back(per_core);
        read_slice(qkv_acc, cb, 0, 0);
        noc.async_read_barrier();
        cb.push_back(per_core);
    }

    // ---- K rows from the ring and from the diagonal taps --------------------
    auto k_rows = [&](const auto& acc, uint32_t cb_id) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(K * per_core);
        for (uint32_t j = 0; j < K; j++) {
            read_slice(acc, cb, j, j * per_core);
        }
        noc.async_read_barrier();
        cb.push_back(K * per_core);
    };
    k_rows(st_acc, cb_state);

    // The shifted ring for the writer: row j <- state row j+1 for j < K-1, row K-1 <- the new input.
    // Filled here (re-reading those rows) so cb_state has ONE consumer: the compute kernel. When the
    // writer shared cb_state instead, compute's pop drained it and the writer blocked forever.
    {
        CircularBuffer cb(cb_shift);
        cb.reserve_back(K * per_core);
        for (uint32_t j = 0; j < K; j++) {
            const bool from_state = (j + 1) < K;
            const uint32_t row = from_state ? (j + 1) : 0;
            uint32_t d = j * per_core;
            for (uint32_t i = 0; i < kd_t; i++) {
                noc.async_read(from_state ? st_acc : qkv_acc, cb, tb,
                               {.page_id = from_state ? (row * c_t + q_col + i) : (q_col + i)},
                               {.offset_bytes = d++ * tb});
            }
            for (uint32_t i = 0; i < kd_t; i++) {
                noc.async_read(from_state ? st_acc : qkv_acc, cb, tb,
                               {.page_id = from_state ? (row * c_t + k_col + i) : (k_col + i)},
                               {.offset_bytes = d++ * tb});
            }
            for (uint32_t i = 0; i < v_t; i++) {
                noc.async_read(from_state ? st_acc : qkv_acc, cb, tb,
                               {.page_id = from_state ? (row * c_t + v_col + i) : (v_col + i)},
                               {.offset_bytes = d++ * tb});
            }
        }
        noc.async_read_barrier();
        cb.push_back(K * per_core);
    }

    // The diagonals are [K, c_t, 32, 32]: one real 32x32 tile per channel, so tile (row j, channel ch)
    // is at page j*c_t + ch and each read is a whole tile (unlike state, which packs 32 channels in
    // one tile's row 0).
    {
        CircularBuffer cb(cb_diag);
        cb.reserve_back(K * per_core);
        for (uint32_t j = 0; j < K; j++) {
            uint32_t d = j * per_core;
            for (uint32_t i = 0; i < kd_t; i++) {
                noc.async_read(di_acc, cb, tb, {.page_id = j * c_t + q_col + i}, {.offset_bytes = d++ * tb});
            }
            for (uint32_t i = 0; i < kd_t; i++) {
                noc.async_read(di_acc, cb, tb, {.page_id = j * c_t + k_col + i}, {.offset_bytes = d++ * tb});
            }
            for (uint32_t i = 0; i < v_t; i++) {
                noc.async_read(di_acc, cb, tb, {.page_id = j * c_t + v_col + i}, {.offset_bytes = d++ * tb});
            }
        }
        noc.async_read_barrier();
        cb.push_back(K * per_core);
    }
}
