// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar BFD re-architecture POC kernel. Copies three input DFBs to one output DFB
// the way real kernels do: init once per operand, then a block loop of tiles.
//
// unary_op_init_common(in0, out) does the full first-time setup (packer for out,
// unpacker + BFD for in0). Switching to in1/in2 uses copy_tile_to_dst_init_short,
// which re-runs llk_unpack_A_init -> programs a fresh unpack BFD from the new input's
// own L1 address (get_local_dfb_interface(operand).tc_slots[0].base_addr). copy_tile
// itself only executes -- it does not program a descriptor -- so the per-operand
// re-init is what repoints the single Unp0 engine at each input.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {
// Copy a block of n_tiles from one input DFB into the output DFB. The caller must have
// programmed the unpack BFD for `in` (via unary_op_init_common or copy_tile_to_dst_init_short)
// before calling this.
void copy_block(DataflowBuffer& in, std::uint32_t in_id, DataflowBuffer& out, std::uint32_t n_tiles) {
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
    constexpr std::uint32_t num_inputs = 3;
    constexpr std::uint32_t tiles_per_input = num_cycles / num_inputs;

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out(dfb::out);

    // in0: full common init (packer for out + unpacker/BFD for in0), then block copy.
    unary_op_init_common(dfb::in0, dfb::out);
    copy_block(dfb_in0, dfb::in0, dfb_out, tiles_per_input);

    // in1: short re-init repoints the unpack BFD to in1's L1 address, then block copy.
    copy_tile_to_dst_init_short(dfb::in1);
    copy_block(dfb_in1, dfb::in1, dfb_out, tiles_per_input);

    // in2: same.
    copy_tile_to_dst_init_short(dfb::in2);
    copy_block(dfb_in2, dfb::in2, dfb_out, tiles_per_input);
}
