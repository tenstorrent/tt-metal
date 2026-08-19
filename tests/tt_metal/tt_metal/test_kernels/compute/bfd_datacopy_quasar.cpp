// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar BFD re-architecture POC kernel. Copies three input DFBs to one output DFB the
// realistic way -- init once per operand, then a block loop of tiles -- and wraps the whole
// three-operand sequence in an outer loop so the unpack partition wraps within a single run.
//
// compute_kernel_hw_startup(in0, out) does the one-time setup (packer for out + unpack/pack
// hw_configure) once, outside every loop. Each operand switch then calls
// copy_tile_to_dst_init_short, which re-runs llk_unpack_A_init -> bump-allocates the next id
// from the unpack partition ([0,16)) and programs it from that input's own L1 address
// (get_local_dfb_interface(operand).tc_slots[0].base_addr). copy_tile itself only executes; it
// programs no descriptor, so there is NO per-tile re-init -- the per-operand init_short is what
// repoints the single Unp0 engine at each input. With num_inputs * num_loops (= 30) operand
// inits per run the partition wraps and hands out reused ids; every copy (including post-wrap
// ones) is bit-exact-checked on the host, so a wrapped id that programs the wrong descriptor is
// caught. The readers re-stream the same tiles each loop (the Quasar tile-counter model consumes
// a tile per unpack, so re-copying requires re-streaming).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {
// Copy a block of n_tiles from one input DFB into the output DFB. The caller must have programmed
// the unpack BFD for `in` (via copy_tile_to_dst_init_short) before calling this. No per-tile
// re-init: the single init before the block covers every tile in it.
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
    constexpr std::uint32_t num_loops = get_arg(args::num_loops);
    constexpr std::uint32_t num_inputs = 3;
    constexpr std::uint32_t tiles_per_input = num_cycles / num_inputs;

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out(dfb::out);

    // One-time hardware setup (packer for out + unpack/pack hw_configure). Done ONCE, outside all
    // loops -- the per-operand copy_tile_to_dst_init_short below programs each unpack BFD.
    compute_kernel_hw_startup(dfb::in0, dfb::out);

    // Outer loop: repeat the whole three-input block copy num_loops times. Each operand switch
    // bump-allocates a fresh unpack BFD, wrapping the partition once its 16 ids are exhausted.
    for (std::uint32_t loop = 0; loop < num_loops; ++loop) {
        copy_tile_to_dst_init_short(dfb::in0);
        copy_block(dfb_in0, dfb::in0, dfb_out, tiles_per_input);

        copy_tile_to_dst_init_short(dfb::in1);
        copy_block(dfb_in1, dfb::in1, dfb_out, tiles_per_input);

        copy_tile_to_dst_init_short(dfb::in2);
        copy_block(dfb_in2, dfb::in2, dfb_out, tiles_per_input);
    }
}
