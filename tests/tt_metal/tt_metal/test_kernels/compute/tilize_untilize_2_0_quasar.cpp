// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 tilize / untilize with only the id-free 2.0 compute API.
// Compile-time args: op (0 = tilize, 1 = untilize), rows (tile rows), block, full (row width in tiles).
//   * tilize: per tile row, tilize_block of `block` tiles from a row-major in DFB (rows `block` tiles wide) to out.
//     tilize_block self-syncs Dest.
//   * untilize: per tile row, per column block: copy_block of `block` tiles into Dest, then pack_untilize_dest_block
//     packs them to the row-major out DFB (rows `full` tiles wide) in the kernel's Dest section.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack_untilize.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/compute/experimental/2_0/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

void kernel_main() {
    using In = LLKOperandFrom<dfb::in0>;
    using Out = LLKOperandFrom<dfb::out0>;
    constexpr std::uint32_t op = get_arg(args::op);
    constexpr std::uint32_t rows = get_arg(args::rows);
    constexpr std::uint32_t block = get_arg(args::block);
    constexpr std::uint32_t full = get_arg(args::full);
    static_assert(full % block == 0, "no tails: block must divide the row width");

    DataflowBuffer in(dfb::in0);
    DataflowBuffer out(dfb::out0);
    ckernel::compute_kernel_hw_startup(in.template front<In>(), out.template back<Out>());

    if constexpr (op == 0) {
        ex::tilize_init<block>(in.template front<In>(), out.template back<Out>());
        for (std::uint32_t r = 0; r < rows; ++r) {
            in.wait_front(block);
            out.reserve_back(block);
            ex::tilize_block<block>(in.template front<In>(), out.template back<Out>());
            out.push_back(block);
            in.pop_front(block);
        }
    } else {
        ex::copy_src_to_dest_init(in.template front<In>());
        ex::pack_untilize_dest_init<block, full>(out.template back<Out>());
        for (std::uint32_t r = 0; r < rows; ++r) {
            out.reserve_back(full);
            for (std::uint32_t c = 0; c < full / block; ++c) {
                in.wait_front(block);
                ckernel::tile_regs_acquire();
                ex::copy_src_to_dest_block<block, 1>(in.template front<In>(), 0);
                ckernel::tile_regs_commit();
                ckernel::tile_regs_wait();
                ex::pack_untilize_dest_block<block, full>(out.template back<Out>(), c);
                ckernel::tile_regs_release();
                in.pop_front(block);
            }
            out.push_back(full);
        }
    }
}
