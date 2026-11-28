// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader for delta_solve_compute.cpp. Per matrix m of this core: A's 16 tiles, alpha as 4 column-broadcast tiles
// (HAS_ALPHA), then B's 16 tiles. A and B are `[..., 128, 128]` fp32 tiled, so matrix m is pages 16 m .. 16 m + 15;
// alpha is `[..., 1, 128]` fp32 tiled (row 0 of 4 tiles per matrix, pages 4 m .. 4 m + 3).
//
// Compile-time args: has_alpha, then the tensor accessor args of a, b, alpha, constants.
// Runtime args: a, b, alpha, constants addresses, first matrix, number of matrices.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_a = 0;
constexpr uint32_t cb_b = 1;
constexpr uint32_t cb_alpha = 2;
constexpr uint32_t cb_const = 3;
constexpr uint32_t cb_scratch = 4;
constexpr uint32_t TILES = 16;
constexpr uint32_t NT = 4;
}  // namespace

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t alpha_addr = get_arg_val<uint32_t>(2);
    const uint32_t const_addr = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    constexpr bool has_alpha = get_compile_time_arg_val(0) != 0;
    constexpr auto a_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto alpha_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto const_args = TensorAccessorArgs<alpha_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer a_buf(cb_a);
    DataflowBuffer b_buf(cb_b);
    DataflowBuffer alpha_buf(cb_alpha);
    DataflowBuffer const_buf(cb_const);
    DataflowBuffer scratch(cb_scratch);
    const uint32_t tile_bytes = a_buf.get_entry_size();
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    const auto alpha = TensorAccessor(alpha_args, alpha_addr);
    const auto constants = TensorAccessor(const_args, const_addr);

    const_buf.reserve_back(3);
    for (uint32_t t = 0; t < 3; ++t) {
        noc.async_read(constants, const_buf, tile_bytes, {.page_id = t}, {.offset_bytes = t * tile_bytes});
    }
    noc.async_read_barrier();
    const_buf.push_back(3);

    const auto stream = [&](DataflowBuffer& buf, const auto& src, uint32_t first_page) {
        for (uint32_t t = 0; t < TILES; ++t) {
            buf.reserve_back(1);
            noc.async_read(src, buf, tile_bytes, {.page_id = first_page + t}, {.offset_bytes = 0});
            noc.async_read_barrier();
            buf.push_back(1);
        }
    };

    for (uint32_t m = first; m < first + count; ++m) {
        stream(a_buf, a, m * TILES);
        if constexpr (has_alpha) {
            // Tile i of the alpha row holds alpha[32 i + c] in row 0, column c. The broadcast tile for rows 32 i ..
            // 32 i + 31 of the transition holds alpha[32 i + r] at every (r, c). Tiles are four 16x16 faces, row-major
            // within a face: (r, c) is at 256 * (2 (r >= 16) + (c >= 16)) + 16 (r % 16) + c % 16.
            scratch.reserve_back(1);
            auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
            for (uint32_t i = 0; i < NT; ++i) {
                noc.async_read(alpha, scratch, tile_bytes, {.page_id = m * NT + i}, {.offset_bytes = 0});
                noc.async_read_barrier();
                alpha_buf.reserve_back(1);
                auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(alpha_buf.get_write_ptr());
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t value = r < 16 ? row[r] : row[256 + r - 16];
                    const uint32_t base = 256 * (2 * (r >= 16)) + 16 * (r % 16);
                    for (uint32_t c = 0; c < 16; ++c) {
                        out[base + c] = value;
                        out[base + 256 + c] = value;
                    }
                }
                alpha_buf.push_back(1);
            }
        }
        stream(b_buf, b, m * TILES);
    }
}
