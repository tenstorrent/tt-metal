// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Alternating-format copy in the static-state-tracking style — the M1
// benchmark. Two streams of different data formats (values: Float16_b,
// indices: UInt16) alternate through the same SrcA -> DST -> pack path, the
// inner-loop shape of the sort / SDPA kernels.
//
// Where LLK 1.0 pays a reconfig_data_format_srca + copy_tile_to_dst_init_short
// pair and a pack_reconfig_data_format on EVERY half-iteration, the tracked
// state proves that only the operand descriptors change: each format swap
// re-emits the THCON_SEC0 descriptor group (+ zero-flag pair) and the pack
// format group, while the datacopy MOPs, addr-mods, counters and strides are
// compile-time proven still valid and elided. The alternation itself is
// genuine (the formats really differ), so this measures the API-granularity
// delta, not redundant-call elision.

#include <cstddef>
#include <cstdint>

#include "experimental/kernel_args.h"      // get_arg(args::…) — named compile-time args
#include "api/dataflow/dataflow_buffer.h"  // dfb::in0/in1/out0/out1 — DFB logical ids

#include "experiments/static-state-tracking/compute/ops.h"
#include "experiments/static-state-tracking/inc/control.h"

void kernel_main() {
    using namespace sst;
    using namespace sst::compute;
    using namespace sst::tensor;

    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t tiles_per_block = 2;

    using TileA = Tile32x32_Float16_b;  // values stream
    using TileB = Tile32x32_UInt16;     // indices stream

    // Default pack reads the natural tiled DST layout -> Remap=false.
    // SrcB descriptor is established for the index stream (it feeds the
    // zero-flag pairing); the pack descriptor starts on the values stream.
    auto s0 = hw_startup<TileA, TileB, TileA, /*Remap=*/false>();

    loop(s0, static_cast<std::size_t>(num_iters), [&](auto s_it, std::size_t /*it*/) {
        // Stream A (Float16_b values). At the loop fixed point the incoming
        // state holds the UInt16 descriptor from stream B — the format swap
        // re-emits the descriptor groups only.
        auto in_a = Tensor<TileA, Dfb>::wait_front(dfb::in0, tiles_per_block);
        sst::compute::tile_regs_acquire();
        auto s_a0 = copy_tile(s_it, in_a, 0, 0);
        auto s_a1 = copy_tile(s_a0, in_a, 1, 1);
        sst::compute::tile_regs_commit();

        sst::compute::tile_regs_wait();
        auto out_a = Tensor<TileA, Dfb>::reserve_back(dfb::out0, tiles_per_block);
        auto s_a2 = pack_tile(s_a1, out_a, 0, 0);
        auto s_a3 = pack_tile(s_a2, out_a, 1, 1);
        sst::compute::tile_regs_release();
        pop_front(in_a);
        push_back(out_a);

        // Stream B (UInt16 indices) — swap back.
        auto in_b = Tensor<TileB, Dfb>::wait_front(dfb::in1, tiles_per_block);
        sst::compute::tile_regs_acquire();
        auto s_b0 = copy_tile(s_a3, in_b, 0, 0);
        auto s_b1 = copy_tile(s_b0, in_b, 1, 1);
        sst::compute::tile_regs_commit();

        sst::compute::tile_regs_wait();
        auto out_b = Tensor<TileB, Dfb>::reserve_back(dfb::out1, tiles_per_block);
        auto s_b2 = pack_tile(s_b1, out_b, 0, 0);
        auto s_b3 = pack_tile(s_b2, out_b, 1, 1);
        sst::compute::tile_regs_release();
        pop_front(in_b);
        push_back(out_b);

        return s_b3;
    });
}
