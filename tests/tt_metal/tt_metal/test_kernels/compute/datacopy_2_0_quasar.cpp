// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 datacopy: in DFB -> copy (FPU datacopy or unpack-to-dest) -> Dest -> pack -> out DFB, using only the
// id-free 2.0 compute API and the take-driven Dest chain. Every op init runs before its op, every Dest section is
// dest_acquire .. dest_release, and the copy/pack TDMA sits between each wait/reserve and its pop/push.
//
// Compile-time args: num_tiles, tiles_per_section (full sections use copy_block<T, 1>, a shorter tail the runtime
// copy_block), to_dest (path), mode: 0 = one run, 6 = chain switch (to-dest 3 sections, FPU datacopy 5, to-dest 3),
// 7 = two DFB pairs (in0 -> out0 then in1 -> out1, num_tiles / 2 each).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/dest_sync.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

constexpr std::uint32_t TPS = get_arg(args::tiles_per_section);

template <bool to_dest, typename In, typename Out, typename Dfb>
void copy_sections(Dfb& in, Dfb& out, std::uint32_t num_tiles) {
    ex::copy_init<to_dest>(in.template front<In>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < num_tiles;) {
        const std::uint32_t n = num_tiles - done < TPS ? num_tiles - done : TPS;
        in.wait_front(n);
        out.reserve_back(n);
        ex::dest_acquire();
        if (n == TPS) {
            ex::copy_block<TPS, 1, to_dest>(in.template front<In>(), 0);
            ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        } else {
            ex::copy_block<to_dest>(in.template front<In>(), 0, 0, n);
            ex::pack_block(out.template back<Out>(), 0, 0, n);
        }
        ex::dest_release();
        out.push_back(n);
        in.pop_front(n);
        done += n;
    }
}

void kernel_main() {
    using In0 = LLKOperandFrom<dfb::in0>;
    using Out0 = LLKOperandFrom<dfb::out0>;
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr bool to_dest = get_arg(args::to_dest) != 0;
    constexpr std::uint32_t mode = get_arg(args::mode);

    DataflowBuffer in0(dfb::in0);
    DataflowBuffer out0(dfb::out0);
    ckernel::compute_kernel_hw_startup(in0.template front<In0>(), out0.template back<Out0>());

    if constexpr (mode == 6) {
        copy_sections<true, In0, Out0>(in0, out0, 3 * TPS);
        copy_sections<false, In0, Out0>(in0, out0, 5 * TPS);
        copy_sections<true, In0, Out0>(in0, out0, 3 * TPS);
    } else if constexpr (mode == 7) {
#if NUM_IN > 1 && NUM_OUT > 1
        using In1 = LLKOperandFrom<dfb::in1>;
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer in1(dfb::in1);
        DataflowBuffer out1(dfb::out1);
        copy_sections<to_dest, In0, Out0>(in0, out0, num_tiles / 2);
        copy_sections<to_dest, In1, Out1>(in1, out1, num_tiles / 2);
#endif
    } else {
        copy_sections<to_dest, In0, Out0>(in0, out0, num_tiles);
    }
}
