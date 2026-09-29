// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "api/compute/eltwise_binary_sfpu.h"

// The original CB views remain FPU-compatible. Each FP32 accumulator also has
// an alias of the same SRAM whose UnpackToDestFp32 mode preserves all bits.
// This mirrors the production matmul partial-reload alias implementation.
uint32_t k2_lossless_view(uint32_t cb) {
    if (cb == K2_SUM_A) {
        return K2_SUM_A_ALIAS;
    }
    if (cb == K2_SUM_B) {
        return K2_SUM_B_ALIAS;
    }
    if (cb == K2_OUT_A) {
        return K2_OUT_A_ALIAS;
    }
    if (cb == K2_OUT_B) {
        return K2_OUT_B_ALIAS;
    }
    return cb;
}

void k2_copy(uint32_t cb, uint32_t tile, uint32_t dst) {
    const uint32_t view = k2_lossless_view(cb);
    UNPACK((get_local_cb_interface(view).fifo_rd_ptr = get_local_cb_interface(cb).fifo_rd_ptr));
    reconfig_data_format_srca(view);
    copy_init(view);
    copy_tile(view, tile, dst);
}

void k2_bcast(uint32_t cb, uint32_t tile, uint32_t dst) {
    const uint32_t view = k2_lossless_view(cb);
    UNPACK((get_local_cb_interface(view).fifo_rd_ptr = get_local_cb_interface(cb).fifo_rd_ptr));
    reconfig_data_format_srca(view);
    unary_bcast_init<BroadcastType::COL>(view);
    unary_bcast<BroadcastType::COL>(view, tile, dst);
    unary_bcast_uninit<BroadcastType::COL>(view);
}

void k2_mul_sum_inplace(uint32_t sum, uint32_t alpha, uint32_t rows) {
    CircularBuffer sums(sum), factors(alpha);
    sums.wait_front(rows);
    factors.wait_front(rows);
    pack_reconfig_data_format(sum);
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        k2_copy(sum, 0, 0);
        k2_bcast(alpha, row, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        sums.pop_front(1);
        sums.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, sum);
        tile_regs_release();
        sums.push_back(1);
    }
}

void k2_add_sum_inplace(uint32_t current, uint32_t previous, uint32_t rows) {
    CircularBuffer cur(current), prev(previous);
    cur.wait_front(rows);
    prev.wait_front(rows);
    pack_reconfig_data_format(current);
    for (uint32_t row = 0; row < rows; ++row) {
        tile_regs_acquire();
        k2_copy(current, row, 0);
        k2_copy(previous, row, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, current);
        tile_regs_release();
    }
    cur.pop_front(rows);
    prev.pop_front(rows);
    cur.reserve_back(rows);
    cur.push_back(rows);
}

template <uint32_t rows, uint32_t cols>
void k2_accumulate_output(uint32_t previous, uint32_t alpha, uint32_t current) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer prev(previous), factors(alpha), cur(current);
    prev.wait_front(count);
    factors.wait_front(rows);
    cur.wait_front(count);
    pack_reconfig_data_format(current);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        k2_copy(previous, tile, 0);
        k2_bcast(alpha, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        k2_copy(current, tile, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, current, tile);
        tile_regs_release();
    }
    prev.pop_front(count);
    factors.pop_front(rows);
    cur.pop_front(count);
    cur.reserve_back(count);
    cur.push_back(count);
}

template <uint32_t rows, uint32_t cols>
void k2_normalize_output(uint32_t input, uint32_t reciprocal, uint32_t output) {
    constexpr uint32_t count = rows * cols;
    CircularBuffer in(input), scale(reciprocal), out(output);
    in.wait_front(count);
    scale.wait_front(rows);
    out.reserve_back(count);
    pack_reconfig_data_format(output);
    for (uint32_t tile = 0; tile < count; ++tile) {
        tile_regs_acquire();
        k2_copy(input, tile, 0);
        k2_bcast(reciprocal, tile / cols, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output);
        tile_regs_release();
    }
    in.pop_front(count);
    scale.pop_front(rows);
    out.push_back(count);
}
