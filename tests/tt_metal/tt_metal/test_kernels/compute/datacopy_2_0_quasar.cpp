// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 datacopy: in DFB -> copy (FPU datacopy or unpack-to-dest) -> Dest -> pack -> out DFB, using only the
// id-free 2.0 compute API. Every op init runs before its op, every Dest use is a legacy tile_regs_acquire / commit /
// wait / release section, and the copy/pack TDMA sits between each wait/reserve and its pop/push.
//
// The copy path: copy_direct_to_dest_* (unpack-to-dest) for a Float32 input at 32-bit Dest, copy_src_to_dest_*
// (FPU datacopy) otherwise.
// Compile-time args: num_tiles, tiles_per_section (full sections use the <T, 1> block forms; a shorter tail re-inits
// and runs a second, shorter instantiation), mode: 0 = one run, 6 = three runs of 3, 5, 3 sections, re-init between, 7
// = two DFB pairs (in0 -> out0 then in1 -> out1, num_tiles / 2 each), 12 = early return by runtime arg skip: 1 = before
// compute_kernel_hw_startup, 2 = right after it, 3 = after one copied + packed section, inside a second section after
// its copy; 0 = one run.
// LTA (host define): in0 / out0 are resident L1 tensors read through LocalTensorAccessor, with no DFBs and no DM
// kernels; one run of num_tiles / tiles_per_section full sections (mode is ignored).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

constexpr std::uint32_t TPS = get_arg(args::tiles_per_section);

// Float32 L1 takes the direct path (the gtest runs it at 32-bit Dest only); the 16-bit formats go through SrcA.
template <typename In>
constexpr bool DIRECT = In::descriptor.format == DataFormat::Float32;

template <typename In>
void copy_init(In in) {
    if constexpr (DIRECT<In>) {
        ex::copy_direct_to_dest_init(in);
    } else {
        ex::copy_src_to_dest_init(in);
    }
}

template <std::uint32_t N, typename In>
void copy_block(In in, std::uint32_t start_itile) {
    if constexpr (DIRECT<In>) {
        ex::copy_direct_to_dest_block<N, 1>(in, start_itile);
    } else {
        ex::copy_src_to_dest_block<N, 1>(in, start_itile);
    }
}

// One Dest section of N tiles: in DFB -> copy -> Dest -> pack -> out DFB.
template <std::uint32_t N, typename In, typename Out, typename Dfb>
void copy_section(Dfb& in, Dfb& out) {
    in.wait_front(N);
    out.reserve_back(N);
    ckernel::tile_regs_acquire();
    copy_block<N>(in.template front<In>(), 0);
    ckernel::tile_regs_commit();
    ckernel::tile_regs_wait();
    ex::pack_block<N, 1>(out.template back<Out>(), 0);
    ckernel::tile_regs_release();
    out.push_back(N);
    in.pop_front(N);
}

template <std::uint32_t NumTiles, typename In, typename Out, typename Dfb>
void copy_sections(Dfb& in, Dfb& out) {
    constexpr std::uint32_t tail = NumTiles % TPS;
    copy_init(in.template front<In>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t i = 0; i < NumTiles / TPS; ++i) {
        copy_section<TPS, In, Out>(in, out);
    }
    if constexpr (tail != 0) {  // block ops take no tails: re-init, then a shorter instantiation
        copy_init(in.template front<In>());
        ex::pack_init(out.template back<Out>());
        copy_section<tail, In, Out>(in, out);
    }
}

#ifdef LTA
void kernel_main() {
    using In0 = LLKOperandFrom<tensor::in0>;
    using Out0 = LLKOperandFrom<tensor::out0>;
    const In0 in = LocalTensorAccessor<std::uint16_t>(tensor::in0).operand<In0>();
    const Out0 out = LocalTensorAccessor<std::uint16_t>(tensor::out0).operand<Out0>();

    ckernel::compute_kernel_hw_startup(in, out);
    copy_init(in);
    ex::pack_init(out);
    for (std::uint32_t i = 0; i < get_arg(args::num_tiles); i += TPS) {
        ckernel::tile_regs_acquire();
        copy_block<TPS>(in, i);
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<TPS, 1>(out, 0, i);
        ckernel::tile_regs_release();
    }
}
#else
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
            copy_sections<TPS, In0, Out0>(in0, out0);
            copy_init(in0.template front<In0>());
            in0.wait_front(TPS);
            ckernel::tile_regs_acquire();
            copy_block<TPS>(in0.template front<In0>(), 0);
            in0.pop_front(TPS);
        }
        if (skip == 0) {
            copy_sections<num_tiles, In0, Out0>(in0, out0);
        }
    } else if constexpr (mode == 6) {
        copy_sections<3 * TPS, In0, Out0>(in0, out0);
        copy_sections<5 * TPS, In0, Out0>(in0, out0);
        copy_sections<3 * TPS, In0, Out0>(in0, out0);
    } else if constexpr (mode == 7) {
#if NUM_IN > 1 && NUM_OUT > 1
        using In1 = LLKOperandFrom<dfb::in1>;
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer in1(dfb::in1);
        DataflowBuffer out1(dfb::out1);
        copy_sections<num_tiles / 2, In0, Out0>(in0, out0);
        copy_sections<num_tiles / 2, In1, Out1>(in1, out1);
#endif
    } else {
        copy_sections<num_tiles, In0, Out0>(in0, out0);
    }
}
#endif
