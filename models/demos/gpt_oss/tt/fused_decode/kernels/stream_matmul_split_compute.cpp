// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TEMP experiment (GPT_OSS_DECODE_EXPERTS_SPLIT=1): the routed-expert down stream's matmul (TRISC) over the exact MXFP4
// split weights (fused_decode/mxfp4_split.py). Per output column: one compressed custom_mm over the BFP4 block (cb_w4:
// small tiles + bias tile per routed expert) and one over the BFP2 block (cb_w2: big tiles per expert, a zero tile
// stepping past each bias multiplier), both from the first activation tile into DST 0 (transposed weight tiles, LoFi,
// FP32 accumulation). Otherwise as stream_matmul_compute.cpp.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/compressed_custom_mm.h"
#include "api/compute/pack.h"
#include "split_meta.hpp"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w4 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);  // activation tiles
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t cb_w2 = get_compile_time_arg_val(5);
    constexpr uint32_t kt4 = get_compile_time_arg_val(6);
    constexpr uint32_t kt2 = get_compile_time_arg_val(7);
    constexpr uint32_t n4 = get_compile_time_arg_val(8);
    constexpr uint32_t n2 = get_compile_time_arg_val(9);
    static constexpr auto meta4 = split_meta::cta_array<10, n4>();
    static constexpr auto meta2 = split_meta::cta_array<10 + n4, n2>();
    const uint32_t meta4_addr = reinterpret_cast<uint32_t>(meta4.data());
    const uint32_t meta2_addr = reinterpret_cast<uint32_t>(meta2.data());
#ifdef SPLIT_REST_PKG
    // the inexact ranks' package bfloat4_b tiles (normal orientation) from cb_w5, their activations from cb_in0r
    constexpr uint32_t rp = 10 + n4 + n2;
    constexpr uint32_t cb_w5 = get_compile_time_arg_val(rp);
    constexpr uint32_t cb_in0r = get_compile_time_arg_val(rp + 1);
    constexpr uint32_t kt5 = get_compile_time_arg_val(rp + 2);
    constexpr uint32_t n5 = get_compile_time_arg_val(rp + 3);
    constexpr uint32_t kt_r = get_compile_time_arg_val(rp + 4);  // activation tiles in cb_in0r
    static constexpr auto meta5 = split_meta::cta_array<rp + 5, n5>();
    const uint32_t meta5_addr = reinterpret_cast<uint32_t>(meta5.data());
#endif

    constexpr bool transpose = true;
    constexpr bool split_acc = false;
    constexpr bool dense_packing = false;
    compressed_custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w4, cb_out);

#ifdef SPLIT_IN0B
    // the BFP2 call reads the exact experts' activations without the bias multiplier tiles (stream_down_writer.cpp)
    constexpr uint32_t cb_in0b = SPLIT_IN0B;
    cb_wait_front(cb_in0b, kt2);
#else
    constexpr uint32_t cb_in0b = cb_in0;
#endif
    cb_wait_front(cb_in0, kt);
#ifdef SPLIT_REST_PKG
    cb_wait_front(cb_in0r, kt_r);
#endif
    for (uint32_t c = 0; c < cols; ++c) {
        tile_regs_acquire();
        cb_wait_front(cb_w4, 1);
        compressed_custom_mm_block<false>(cb_in0, cb_w4, meta4_addr, 0, kt4);
        cb_pop_front(cb_w4, 1);
#ifndef SPLIT_NO_BFP2
        cb_wait_front(cb_w2, 1);
        compressed_custom_mm_block<false>(cb_in0b, cb_w2, meta2_addr, 0, kt2);
        cb_pop_front(cb_w2, 1);
#endif
#ifdef SPLIT_REST_PKG
        compressed_custom_mm_block_init_short<false, split_acc, dense_packing>(cb_in0r, cb_w5, cb_out);
        cb_wait_front(cb_w5, 1);
        compressed_custom_mm_block<false>(cb_in0r, cb_w5, meta5_addr, 0, kt5);
        cb_pop_front(cb_w5, 1);
        compressed_custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_in0, cb_w4, cb_out);
#endif
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_in0, kt);
#ifdef SPLIT_REST_PKG
    cb_pop_front(cb_in0r, kt_r);
#endif
#ifdef SPLIT_IN0B
    cb_pop_front(cb_in0b, kt2);
#endif
    compressed_custom_mm_block_uninit<dense_packing>();
}
