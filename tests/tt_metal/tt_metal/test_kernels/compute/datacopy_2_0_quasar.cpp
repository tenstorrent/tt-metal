// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 datacopy: in DFB -> copy (FPU datacopy or unpack-to-dest) -> Dest -> pack -> out DFB, using only the
// id-free 2.0 compute API. Every op init runs before its op, every Dest use is a legacy tile_regs_acquire / commit /
// wait / release section, and the copy/pack TDMA sits between each wait/reserve and its pop/push.
//
// The copy path: unpack-to-dest for a Float32 input at 32-bit Dest, FPU datacopy otherwise.
// Compile-time args: num_tiles, tiles_per_section (full sections use copy_block<T, 1>, a shorter tail re-inits and
// uses the runtime copy_block), mode: 0 = one run, 6 = three runs of 3, 5, 3 sections, re-init between, 7 = two
// DFB pairs (in0 -> out0 then in1 -> out1, num_tiles / 2 each), 12 = early return by runtime arg skip: 1 = before
// compute_kernel_hw_startup, 2 = right after it, 3 = after one copied + packed section, inside a second section after
// its copy; 0 = one run.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

constexpr std::uint32_t TPS = get_arg(args::tiles_per_section);

template <typename In, typename Out, typename Dfb>
void copy_sections(Dfb& in, Dfb& out, std::uint32_t num_tiles) {
    ex::copy_init(in.template front<In>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < num_tiles;) {
        const std::uint32_t n = num_tiles - done < TPS ? num_tiles - done : TPS;
        if (n != TPS) {  // tail: block ops take no tails, so re-init before the shorter block
            ex::copy_init(in.template front<In>());
            ex::pack_init(out.template back<Out>());
        }
        in.wait_front(n);
        out.reserve_back(n);
        ckernel::tile_regs_acquire();
        if (n == TPS) {
            ex::copy_block<TPS, 1>(in.template front<In>(), 0);
        } else {
            ex::copy_block(in.template front<In>(), 0, 0, n);
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        if (n == TPS) {
            ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        } else {
            ex::pack_block(out.template back<Out>(), 0, 0, n);
        }
        ckernel::tile_regs_release();
        out.push_back(n);
        in.pop_front(n);
        done += n;
    }
}

void kernel_main() {
    using In0 = LLKOperandFrom<dfb::in0>;
    using Out0 = LLKOperandFrom<dfb::out0>;
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr std::uint32_t mode = get_arg(args::mode);

    DataflowBuffer in0(dfb::in0);
    DataflowBuffer out0(dfb::out0);
    [[maybe_unused]] const std::uint32_t skip = get_arg(args::skip);
    if constexpr (mode == 12) {
        if (skip == 1) {
            return;
        }
    }
    ckernel::compute_kernel_hw_startup(in0.template front<In0>(), out0.template back<Out0>());

    if constexpr (mode == 12) {
        if (skip == 3) {
            copy_sections<In0, Out0>(in0, out0, TPS);
            ex::copy_init(in0.template front<In0>());
            in0.wait_front(TPS);
            ckernel::tile_regs_acquire();
            ex::copy_block<TPS, 1>(in0.template front<In0>(), 0);
            in0.pop_front(TPS);
        }
        if (skip == 0) {
            copy_sections<In0, Out0>(in0, out0, num_tiles);
        }
    } else if constexpr (mode == 6) {
        copy_sections<In0, Out0>(in0, out0, 3 * TPS);
        copy_sections<In0, Out0>(in0, out0, 5 * TPS);
        copy_sections<In0, Out0>(in0, out0, 3 * TPS);
    } else if constexpr (mode == 7) {
#if NUM_IN > 1 && NUM_OUT > 1
        using In1 = LLKOperandFrom<dfb::in1>;
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer in1(dfb::in1);
        DataflowBuffer out1(dfb::out1);
        copy_sections<In0, Out0>(in0, out0, num_tiles / 2);
        copy_sections<In1, Out1>(in1, out1, num_tiles / 2);
#endif
    } else {
        copy_sections<In0, Out0>(in0, out0, num_tiles);
    }
}
