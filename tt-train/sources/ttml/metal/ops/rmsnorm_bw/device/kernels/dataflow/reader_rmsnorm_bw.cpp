// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for both rmsnorm_bw phases. Work item = (tile-row r, slice s); the slice's tiles
// [s*St, min(Wt, (s+1)*St)) of a / gamma / dL_dout are streamed `block` tiles at a time.
// With APPLY (phase B) the row's rms tile and its S partial-sum tiles are read first.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/rmsnorm_bw_cbs.hpp"

namespace cb = rmsnorm_bw_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t mask_w = get_compile_time_arg_val(4);

// ones_row0[r][c] = (r == 0): row 0 lives in the first 16 elements of faces 0 and 1.
FORCE_INLINE void generate_ones_row0_tile(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024U; ++i) {
        ptr[i] = 0;
    }
    for (uint32_t c = 0; c < 16U; ++c) {
        ptr[c] = 0x3F80;         // face 0, row 0
        ptr[256U + c] = 0x3F80;  // face 1, row 0
    }
    cb_push_back(cb_id, 1);
}

template <typename Accessor>
FORCE_INLINE void read_block_async(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, n);
    uint32_t l1 = get_write_ptr(cb_id);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t a_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dy_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t rms_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t partials_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto a_args = TensorAccessorArgs<5>();
    constexpr auto gamma_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto dy_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    const auto a_acc = TensorAccessor(a_args, a_addr);
    const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
    const auto dy_acc = TensorAccessor(dy_args, dy_addr);
#ifdef APPLY
    constexpr auto rms_args = TensorAccessorArgs<dy_args.next_compile_time_args_offset()>();
    constexpr auto partials_args = TensorAccessorArgs<rms_args.next_compile_time_args_offset()>();
    const auto rms_acc = TensorAccessor(rms_args, rms_addr);
    const auto partials_acc = TensorAccessor(partials_args, partials_addr);
    const uint32_t f32_tile_bytes = get_tile_size(cb::partials);
#endif

    const uint32_t tile_bytes = get_tile_size(cb::a);

    // Constant tiles.
    generate_tile_with_bfloat16_value(cb::zero, 0x0);
#ifdef APPLY
    generate_tile_with_bfloat16_value(cb::ones, 0x3F80);
    generate_ones_row0_tile(cb::ones_row0);
#else
#ifdef DO_MASK_W
    generate_mask_tile(cb::mask, /*fill*/ 0x3F80, /*mask_fill*/ 0x0, mask_w);
#endif
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / S;
        const uint32_t s = work - r * S;
        const uint32_t col0 = s * St;
        const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);

#ifdef APPLY
        // rms is [rows*32, 1] -> one tile per tile-row; partials are [rows*32, S*32] -> S tiles per tile-row.
        read_block_async(cb::rms, rms_acc, r, 1, tile_bytes);
        read_block_async(cb::partials, partials_acc, r * S, S, f32_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb::rms, 1);
        cb_push_back(cb::partials, S);
#endif

        for (uint32_t c = 0; c < ncols; c += block) {
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            const uint32_t page = r * Wt + col0 + c;
            read_block_async(cb::a, a_acc, page, n, tile_bytes);
            read_block_async(cb::gamma, gamma_acc, col0 + c, n, tile_bytes);
            read_block_async(cb::dy, dy_acc, page, n, tile_bytes);
            noc_async_read_barrier();
            cb_push_back(cb::a, n);
            cb_push_back(cb::gamma, n);
            cb_push_back(cb::dy, n);
        }
    }
}
