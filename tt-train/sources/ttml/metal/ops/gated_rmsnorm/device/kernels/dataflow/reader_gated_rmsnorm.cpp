// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/gated_rmsnorm_cbs.hpp"

namespace cb = ttml_gated_rmsnorm_cb;

constexpr uint32_t Gt = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t num_groups = get_compile_time_arg_val(2);

// A bf16 32x32 tile is four 16x16 faces (top-left, top-right, bottom-left, bottom-right) of 128 u32
// words each, 8 words per face row. Copies row 0 of each top face over every row of its column half.
inline void broadcast_gamma_rows(const uint32_t l1_addr) {
    constexpr uint32_t face_words = 128U;
    constexpr uint32_t row_words = 8U;
    constexpr uint32_t face_rows = 16U;
    auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t half = 0; half < 2U; ++half) {
        volatile tt_l1_ptr uint32_t* top = tile + half * face_words;
        volatile tt_l1_ptr uint32_t* bottom = tile + (half + 2U) * face_words;
        for (uint32_t w = 0; w < row_words; ++w) {
            const uint32_t value = top[w];
            for (uint32_t r = 1; r < face_rows; ++r) {
                top[r * row_words + w] = value;
            }
            for (uint32_t r = 0; r < face_rows; ++r) {
                bottom[r * row_words + w] = value;
            }
        }
    }
}

void kernel_main() {
    uint32_t arg = 0U;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gate_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t dy_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto gate_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    const auto x_acc = TensorAccessor(x_args, x_addr);
    const auto gate_acc = TensorAccessor(gate_args, gate_addr);
    const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
#ifdef BACKWARD
    constexpr auto dy_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    const auto dy_acc = TensorAccessor(dy_args, dy_addr);
#endif

    const uint32_t tile_bytes = get_tile_size(cb::x);

    generate_tile_with_bfloat16_value(cb::ones, BF16_ONE_BITS);

    read_tiles_by_row<false>(cb::gamma_b, gamma_acc, 0U, Gt, tile_bytes, Gt);
    noc_async_read_barrier();
    const uint32_t gamma_l1 = get_write_ptr(cb::gamma_b);
    for (uint32_t j = 0; j < Gt; ++j) {
        broadcast_gamma_rows(gamma_l1 + j * tile_bytes);
    }
    cb_push_back(cb::gamma_b, Gt);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / num_groups;
        const uint32_t h = work - r * num_groups;
        const uint32_t first_page = r * Wt + h * Gt;

        read_tiles_by_row<false>(cb::x, x_acc, first_page, Gt, tile_bytes, Gt);
        read_tiles_by_row<false>(cb::gate, gate_acc, first_page, Gt, tile_bytes, Gt);
#ifdef BACKWARD
        read_tiles_by_row<false>(cb::dy, dy_acc, first_page, Gt, tile_bytes, Gt);
#endif
        noc_async_read_barrier();
        cb_push_back(cb::x, Gt);
        cb_push_back(cb::gate, Gt);
#ifdef BACKWARD
        cb_push_back(cb::dy, Gt);
#endif
    }
}
