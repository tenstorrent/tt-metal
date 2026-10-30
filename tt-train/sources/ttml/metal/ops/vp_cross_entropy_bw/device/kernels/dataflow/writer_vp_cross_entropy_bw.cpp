// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// vp_cross_entropy_bw writer: per item (row r, slice s) the row's target page; per output tile the one-hot term
// scale * g is subtracted in L1 at (h, target[h] - first_v) when the target falls into the tile, then the tile is
// written. g is read from the grad tile's first element.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"

namespace generic = norm::kernel_util::generic;

constexpr uint32_t cb_out = 3;
constexpr uint32_t cb_target = 4;
constexpr uint32_t cb_grad = 5;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t Ht = get_compile_time_arg_val(1);
constexpr uint32_t slices = get_compile_time_arg_val(2);
constexpr uint32_t St = get_compile_time_arg_val(3);
constexpr uint32_t block = get_compile_time_arg_val(4);
constexpr uint32_t target_page_bytes = get_compile_time_arg_val(5);
constexpr uint32_t target_read_bytes = get_compile_time_arg_val(6);
constexpr uint32_t scale_bits = get_compile_time_arg_val(7);
constexpr auto out_args = TensorAccessorArgs<8>();
constexpr auto target_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto grad_args = TensorAccessorArgs<target_args.next_compile_time_args_offset()>();

void kernel_main() {
    using namespace tt::constants;
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t target_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t first_v = get_arg_val<uint32_t>(arg++);
    const uint32_t last_v = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t out_tile = get_tile_size(cb_out);
    const uint32_t grad_tile = get_tile_size(cb_grad);
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile);
    const auto target_acc = TensorAccessor(target_args, target_addr, target_page_bytes);
    const auto grad_acc = TensorAccessor(grad_args, grad_addr, grad_tile);

    // The upstream gradient g: element (0, 0) of its tile.
    cb_reserve_back(cb_grad, 1);
    noc_async_read_page(0, grad_acc, get_write_ptr(cb_grad));
    noc_async_read_barrier();
    cb_push_back(cb_grad, 1);
    cb_wait_front(cb_grad, 1);
#ifdef GRAD_FP32
    const float g = *reinterpret_cast<volatile tt_l1_ptr float *>(get_read_ptr(cb_grad));
#else
    const float g = bfloat16_to_float(*reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(cb_grad)));
#endif
    const float one_hot = generic::bit_cast<float>(scale_bits) * g;

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / slices;
        const uint32_t s = work - r * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);

        cb_reserve_back(cb_target, 1);
        read_target_indices_page_clamped(
            target_acc, get_write_ptr(cb_target), r, Ht, target_page_bytes, target_read_bytes);
        cb_push_back(cb_target, 1);
        cb_wait_front(cb_target, 1);
        auto targets = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(cb_target));

        for (uint32_t k = 0; k < nk; k += block) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_wait_front(cb_out, block);
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t wt = k0 + k + j;
                const uint32_t col_start = wt * TILE_WIDTH;
                auto tile = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1);
                for (uint32_t h = 0; h < TILE_HEIGHT; ++h) {
                    const uint32_t c = targets[h];
                    if (c >= first_v && c < last_v) {
                        const uint32_t local_c = c - first_v;
                        if (local_c >= col_start && local_c < col_start + TILE_WIDTH) {
                            const uint32_t idx = get_tilized_idx(h, local_c - col_start);
                            tile[idx] = float_to_bfloat16(bfloat16_to_float(tile[idx]) - one_hot);
                        }
                    }
                }
                noc_async_write_page(r * Wt + wt, out_acc, l1);
                l1 += out_tile;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, block);
        }
        cb_pop_front(cb_target, 1);
    }
}
