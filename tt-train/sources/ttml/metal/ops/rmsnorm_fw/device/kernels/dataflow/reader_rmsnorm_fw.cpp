// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for both rmsnorm_fw phases. Work item = (tile-row r, slice s); the slice's tiles
// [s*St, min(Wt, (s+1)*St)) of x (and gamma with APPLY) are streamed `block` tiles at a time.
// With APPLY (phase B) the row's S partial-sum tiles are read first.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/rmsnorm_fw_cbs.hpp"

namespace cb = rmsnorm_fw_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr uint32_t mask_w = get_compile_time_arg_val(4);

// Streams are always pushed/popped in whole `block`s (a CB only wraps cleanly when every push has the same
// size); a short final block of a slice fills `n` tiles and leaves the rest of the block unused.
template <typename Accessor>
FORCE_INLINE void read_tiles_async(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t reserve, uint32_t tile_bytes) {
    cb_reserve_back(cb_id, reserve);
    uint32_t l1 = get_write_ptr(cb_id);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t partials_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto x_args = TensorAccessorArgs<5>();
    const auto x_acc = TensorAccessor(x_args, x_addr);
#ifdef APPLY
    constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto partials_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
    const auto partials_acc = TensorAccessor(partials_args, partials_addr);
    const uint32_t f32_tile_bytes = get_tile_size(cb::partials);
#endif

    const uint32_t tile_bytes = get_tile_size(cb::x);

    // Constant tiles.
    generate_tile_with_bfloat16_value(cb::zero, 0x0);
#ifdef APPLY
    generate_tile_with_bfloat16_value(cb::ones, 0x3F80);
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
        // partials are [rows*32, S*32] -> S tiles per tile-row.
        read_tiles_async(cb::partials, partials_acc, r * S, S, S, f32_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb::partials, S);
#endif

        for (uint32_t c = 0; c < ncols; c += block) {
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            const uint32_t page = r * Wt + col0 + c;
            read_tiles_async(cb::x, x_acc, page, n, block, tile_bytes);
#ifdef APPLY
            read_tiles_async(cb::gamma, gamma_acc, col0 + c, n, block, tile_bytes);
#endif
            noc_async_read_barrier();
            cb_push_back(cb::x, block);
#ifdef APPLY
            cb_push_back(cb::gamma, block);
#endif
        }
    }
}
