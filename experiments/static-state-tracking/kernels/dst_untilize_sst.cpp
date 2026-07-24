// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dst_untilize — real Blackhole compute kernel, written from scratch in the
// static-state-tracking style. JIT-compiled and launched by
//   tests/tt_metal/tt_metal/llk/test_untilize_tilize.cpp
// (TensixComputePackUntilizeDstStaticState), verified against the untilize
// golden.

#include <cstddef>
#include <cstdint>

#include "experimental/kernel_args.h"      // get_arg(args::…) — named compile-time args
#include "api/dataflow/dataflow_buffer.h"  // dfb::in / dfb::out — DFB logical ids

#include "experiments/static-state-tracking/compute/ops.h"
#include "experiments/static-state-tracking/inc/control.h"

namespace {

// Mirrors compute_num_blocks_per_col() from the reference dst_untilize.cpp:
// pick the largest block width (<= DST half/full capacity) that divides the row.
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt, bool fp32_dest_acc) {
    const uint32_t max_bct = fp32_dest_acc ? 4u : 8u;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }
    return 1u;
}

}  // namespace

void kernel_main() {
    using namespace sst;
    using namespace sst::compute;
    using namespace sst::tensor;

    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);            // num_tiles_r
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);  // num_tiles_c

    constexpr bool fp32 = (DST_ACCUM_MODE != 0);
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(per_core_block_tile_cnt, fp32);
    constexpr uint16_t tiles_per_block = static_cast<uint16_t>(per_core_block_tile_cnt / num_blocks_per_col);
    constexpr uint16_t tiles_per_row = static_cast<uint16_t>(per_core_block_tile_cnt);

    using TileT = Tile32x32_Float16_b;

    auto s0 = hw_startup<TileT, TileT, TileT>();

    // Outer loop over block-rows. reserve_back / push_back are pure dataflow and
    // carry no compute state, so they sit in the body around the tracked ops.
    loop(s0, static_cast<std::size_t>(per_core_block_cnt), [&](auto s_r, std::size_t /*r*/) {
        auto out = Tensor<TileT, Dfb>::reserve_back(dfb::out, tiles_per_row);

        auto s_row_end = loop(s_r, static_cast<std::size_t>(num_blocks_per_col), [&](auto s_b, std::size_t b) {
            auto in = Tensor<TileT, Dfb>::wait_front(dfb::in, tiles_per_block);

            sst::compute::tile_regs_acquire();
            // Inner loop: copy tiles_per_block tiles A -> DST[0..tiles_per_block-1].
            auto s_copied = loop(s_b, static_cast<std::size_t>(tiles_per_block), [&](auto s, std::size_t i) {
                return copy_tile(s, in, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
            });
            sst::compute::tile_regs_commit();

            sst::compute::tile_regs_wait();
            // Drain the DST block to L1 in untilized (row-major) layout.
            auto s_packed = untilize_block<tiles_per_block, tiles_per_row>(
                s_copied, out, static_cast<uint32_t>(b) * tiles_per_block);
            sst::compute::tile_regs_release();

            pop_front(in);
            return s_packed;
        });

        push_back(out);
        return s_row_end;
    });
}
