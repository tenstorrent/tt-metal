// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for both add_rmsnorm_fw phases. Work item = (tile-row r, slice s); the slice's tiles
// [s*St, min(Wt, (s+1)*St)) are streamed `block` tiles at a time.
//   phase A: residual and delta tiles
//   phase B (APPLY): the row's S partial tiles first, then sum and gamma tiles

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/device/kernels/add_rmsnorm_fw_allcore_cbs.hpp"

namespace cb = add_rmsnorm_fw_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);

// Reserves `reserve` tiles and fills the first `n`: a CB push must never straddle the end of the buffer (the
// producer writes contiguous memory and only an exact wrap is legal), so the streamed CBs always move in full
// `block` pushes and the shorter tail of a slice leaves the rest of its block stale.
template <typename Accessor>
FORCE_INLINE void read_block_async(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t tile_bytes, uint32_t reserve) {
    cb_reserve_back(cb_id, reserve);
    uint32_t l1 = get_write_ptr(cb_id);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t in1_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t in2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto in0_args = TensorAccessorArgs<4>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in0_acc = TensorAccessor(in0_args, in0_addr);  // residual (A) / sum (B)
    const auto in1_acc = TensorAccessor(in1_args, in1_addr);  // delta (A) / gamma (B)
#ifdef APPLY
    constexpr auto in2_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    const auto partials_acc = TensorAccessor(in2_args, in2_addr);
    const uint32_t f32_tile_bytes = get_tile_size(cb::partials);
    constexpr uint32_t cb_in0 = cb::sum;
    constexpr uint32_t cb_in1 = cb::gamma;
    generate_tile_with_bfloat16_value(cb::ones, 0x3F80);
#else
    constexpr uint32_t cb_in0 = cb::residual;
    constexpr uint32_t cb_in1 = cb::delta;
#endif
    generate_tile_with_bfloat16_value(cb::zero, 0x0);

    const uint32_t tile_bytes = get_tile_size(cb_in0);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / S;
        const uint32_t s = work - r * S;
        const uint32_t col0 = s * St;
        const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);

#ifdef APPLY
        // partials are [rows*32, S*32] -> S tiles per tile-row.
        read_block_async(cb::partials, partials_acc, r * S, S, f32_tile_bytes, S);
        noc_async_read_barrier();
        cb_push_back(cb::partials, S);
#endif

        for (uint32_t c = 0; c < ncols; c += block) {
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            const uint32_t page = r * Wt + col0 + c;
            read_block_async(cb_in0, in0_acc, page, n, tile_bytes, block);
#ifdef APPLY
            read_block_async(cb_in1, in1_acc, col0 + c, n, tile_bytes, block);  // gamma: one tile-row
#else
            read_block_async(cb_in1, in1_acc, page, n, tile_bytes, block);
#endif
            noc_async_read_barrier();
            cb_push_back(cb_in0, block);
            cb_push_back(cb_in1, block);
        }
    }
}
