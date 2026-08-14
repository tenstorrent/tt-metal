// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dst_untilize with GREEDY blocking: fill DST to capacity ([cap, cap, ...,
// remainder] blocks per row) regardless of whether the block width divides the
// row width.
//
// The LLK 1.0 pack-untilize API cannot express this: _llk_pack_untilize_init_
// statically requires full_ct_dim % block_ct_dim == 0 and addresses blocks by
// an aligned block_c_index, so the production dst_untilize kernel must pick
// the largest DIVISOR of the row width as its block width
// (compute_num_blocks_per_col). For a 22-tile row that means 11 DST
// round-trips of 2 tiles; greedy blocking does 3 ([8, 8, 6]). Each DST round
// is a full MATH/PACK tile_regs handshake, so the round count is pipeline
// structure, not just config overhead.
//
// In the SST model the row stride (keyed on the row width) and the PACR MOP
// (keyed on the block width) are separate tracked fields, and the untilize op
// takes an explicit tile offset — so mixed-width rows are expressible, the
// remainder block costs one extra MOP configure, and every other configure is
// compile-time elided.

#include <cstddef>
#include <cstdint>

#include "experimental/kernel_args.h"      // get_arg(args::…) — named compile-time args
#include "api/dataflow/dataflow_buffer.h"  // dfb::in / dfb::out — DFB logical ids

#include "experiments/static-state-tracking/compute/ops.h"
#include "experiments/static-state-tracking/inc/control.h"

void kernel_main() {
    using namespace sst;
    using namespace sst::compute;
    using namespace sst::tensor;

    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);            // rows
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);  // row width, in tiles

    constexpr bool fp32 = (DST_ACCUM_MODE != 0);
    constexpr uint16_t dst_cap = fp32 ? 4 : 8;
    constexpr uint32_t num_full = per_core_block_tile_cnt / dst_cap;
    constexpr uint16_t w_rem = static_cast<uint16_t>(per_core_block_tile_cnt % dst_cap);
    constexpr uint16_t full_ct_dim = static_cast<uint16_t>(per_core_block_tile_cnt);
    static_assert(num_full > 0, "row width must be at least the DST capacity for the greedy benchmark");

    using TileT = Tile32x32_Float16_b;

    auto s0 = hw_startup<TileT, TileT, TileT>();

    loop(s0, static_cast<std::size_t>(per_core_block_cnt), [&](auto s_r, std::size_t /*r*/) {
        auto out = Tensor<TileT, Dfb>::reserve_back(dfb::out, full_ct_dim);

        // Full-capacity blocks. All configuration is emitted at most once per
        // row (only when the previous block width differed) and elided at the
        // inner fixed point.
        auto s_main = loop(s_r, static_cast<std::size_t>(num_full), [&](auto s_b, std::size_t b) {
            auto in = Tensor<TileT, Dfb>::wait_front(dfb::in, dst_cap);

            sst::compute::tile_regs_acquire();
            auto s_copied = loop(s_b, static_cast<std::size_t>(dst_cap), [&](auto s, std::size_t i) {
                return copy_tile(s, in, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
            });
            sst::compute::tile_regs_commit();

            sst::compute::tile_regs_wait();
            auto s_packed = untilize_block<dst_cap, full_ct_dim>(s_copied, out, static_cast<uint32_t>(b) * dst_cap);
            sst::compute::tile_regs_release();

            pop_front(in);
            return s_packed;
        });

        // Remainder block (width w_rem < dst_cap): one extra MOP configure per
        // row; row strides / descriptors / dest state proven unchanged.
        if constexpr (w_rem > 0) {
            auto in = Tensor<TileT, Dfb>::wait_front(dfb::in, w_rem);

            sst::compute::tile_regs_acquire();
            auto s_copied = loop(s_main, static_cast<std::size_t>(w_rem), [&](auto s, std::size_t i) {
                return copy_tile(s, in, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
            });
            sst::compute::tile_regs_commit();

            sst::compute::tile_regs_wait();
            auto s_packed = untilize_block<w_rem, full_ct_dim>(s_copied, out, num_full * dst_cap);
            sst::compute::tile_regs_release();

            pop_front(in);
            push_back(out);
            return s_packed;
        } else {
            push_back(out);
            return s_main;
        }
    });
}
