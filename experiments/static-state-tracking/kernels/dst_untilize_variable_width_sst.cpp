// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dst_untilize with a VARIABLE block width per row, in the static-state-
// tracking style — the granularity benchmark. Each output row is packed as
// (width/w_main - 1) blocks of w_main tiles plus two half-width tail blocks,
// the real shape of untilize kernels with mixed block widths (matmul
// untilize-out, transpose_wh_rm, pool compute kernels re-init pack per block
// the same way).
//
// Where the LLK 1.0 kernel must re-run pack_untilize_dest_init on every width
// switch (pack format reconfig + untilize init + dest-offset registers, per
// switch, per row), the tracked state proves that only the untilize PACR MOP
// depends on the block width: the width switch re-emits ONE sub-step
// (pack_untilize_mop_cfg) and the row strides / dest-offset / format state are
// compile-time proven still valid and elided. Same steady-state MOPs, less
// per-switch configuration.

#include <cstddef>
#include <cstdint>

#include "experimental/kernel_args.h"      // get_arg(args::…) — named compile-time args
#include "api/dataflow/dataflow_buffer.h"  // dfb::in / dfb::out — DFB logical ids

#include "experiments/static-state-tracking/compute/ops.h"
#include "experiments/static-state-tracking/inc/control.h"

namespace {

constexpr uint32_t largest_even_divisor_leq(uint32_t n, uint32_t cap) {
    for (uint32_t d = cap; d >= 2; --d) {
        if ((d % 2 == 0) && (n % d == 0)) {
            return d;
        }
    }
    return 0;
}

}  // namespace

void kernel_main() {
    using namespace sst;
    using namespace sst::compute;
    using namespace sst::tensor;

    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);            // rows
    constexpr uint32_t per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);  // row width, in tiles

    constexpr bool fp32 = (DST_ACCUM_MODE != 0);
    constexpr uint32_t dst_cap = fp32 ? 4 : 8;
    constexpr uint16_t w_main = static_cast<uint16_t>(largest_even_divisor_leq(per_core_block_tile_cnt, dst_cap));
    static_assert(w_main >= 2, "row width needs an even divisor <= DST capacity");
    constexpr uint16_t w_tail = w_main / 2;
    constexpr uint32_t num_main = per_core_block_tile_cnt / w_main - 1;  // last main block split into two tails
    constexpr uint32_t num_tail = 2;
    constexpr uint16_t full_ct_dim = static_cast<uint16_t>(per_core_block_tile_cnt);
    static_assert(num_main > 0, "variable-width benchmark wants at least one full block plus the tail pair");

    using TileT = Tile32x32_Float16_b;

    auto s0 = hw_startup<TileT, TileT, TileT>();

    loop(s0, static_cast<std::size_t>(per_core_block_cnt), [&](auto s_r, std::size_t /*r*/) {
        auto out = Tensor<TileT, Dfb>::reserve_back(dfb::out, full_ct_dim);

        // Main blocks (width w_main). Block 0 of each row re-emits the untilize
        // MOP (the tracked width is w_tail from the previous row); blocks 1..
        // run at the inner fixed point with every configure compiled out.
        auto s_main = loop(s_r, static_cast<std::size_t>(num_main), [&](auto s_b, std::size_t b) {
            auto in = Tensor<TileT, Dfb>::wait_front(dfb::in, w_main);

            sst::compute::tile_regs_acquire();
            auto s_copied = loop(s_b, static_cast<std::size_t>(w_main), [&](auto s, std::size_t i) {
                return copy_tile(s, in, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
            });
            sst::compute::tile_regs_commit();

            sst::compute::tile_regs_wait();
            auto s_packed = untilize_block<w_main, full_ct_dim>(s_copied, out, static_cast<uint32_t>(b) * w_main);
            sst::compute::tile_regs_release();

            pop_front(in);
            return s_packed;
        });

        // Tail blocks (width w_tail = w_main/2): the width switch re-emits only
        // the MOP sub-step on the first tail block; row strides and dest state
        // are proven unchanged and elided, and the second tail block runs at
        // the fixed point with zero configuration.
        auto s_tail = loop(s_main, static_cast<std::size_t>(num_tail), [&](auto s_t, std::size_t t) {
            auto in = Tensor<TileT, Dfb>::wait_front(dfb::in, w_tail);

            sst::compute::tile_regs_acquire();
            auto s_copied = loop(s_t, static_cast<std::size_t>(w_tail), [&](auto s, std::size_t i) {
                return copy_tile(s, in, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
            });
            sst::compute::tile_regs_commit();

            sst::compute::tile_regs_wait();
            auto s_packed = untilize_block<w_tail, full_ct_dim>(
                s_copied, out, num_main * w_main + static_cast<uint32_t>(t) * w_tail);
            sst::compute::tile_regs_release();

            pop_front(in);
            return s_packed;
        });

        push_back(out);
        return s_tail;
    });
}
