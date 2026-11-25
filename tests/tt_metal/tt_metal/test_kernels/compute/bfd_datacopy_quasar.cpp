// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Copies three input DFBs to one output DFB (copy_init per operand, then a block of tiles), looped so the
// fixed Unp0 BFD is re-programmed per operand; the host checks every copy bit-exact.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {
// Copy a block of n_tiles from one input DFB into the output DFB. The caller must have programmed
// the unpack BFD for `in` (via copy_init) before calling this. No per-tile
// re-init: the single init before the block covers every tile in it.
template <typename DFBIn, typename DFBOut>
void copy_block(DFBIn& in, std::uint32_t in_id, DFBOut& out, std::uint32_t n_tiles) {
    for (std::uint32_t t = 0; t < n_tiles; ++t) {
        in.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        tile_regs_wait();
        copy_tile(in_id, 0, 0);
        pack_tile(0, dfb::out);
        tile_regs_commit();
        tile_regs_release();
        in.pop_front(1);
        out.push_back(1);
    }
}
}  // namespace

void kernel_main() {
    constexpr std::uint32_t num_cycles = get_arg(args::num_cycles);
    constexpr std::uint32_t num_loops = get_arg(args::num_loops);
    constexpr std::uint32_t num_inputs = 3;
    constexpr std::uint32_t tiles_per_input = num_cycles / num_inputs;

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out(dfb::out);

    // One-time hardware setup (packer for out + unpack/pack hw_configure). Done ONCE, outside all
    // loops -- the per-operand copy_init below programs each unpack BFD.
    compute_kernel_hw_startup(dfb::in0, dfb::out);

    // Outer loop: repeat the whole three-input block copy num_loops times. Each operand switch
    // re-programs the Unp0 BFD.
    for (std::uint32_t loop = 0; loop < num_loops; ++loop) {
        copy_init(dfb::in0);
        copy_block(dfb_in0, dfb::in0, dfb_out, tiles_per_input);

        DPRINT("after copy_block in0\n");

        copy_init(dfb::in1);
        copy_block(dfb_in1, dfb::in1, dfb_out, tiles_per_input);

        DPRINT("after copy_block in1\n");

        copy_init(dfb::in2);
        copy_block(dfb_in2, dfb::in2, dfb_out, tiles_per_input);

        DPRINT("after copy_block in2\n");
    }
}
