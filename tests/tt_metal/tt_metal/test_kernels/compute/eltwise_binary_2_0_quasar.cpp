// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 eltwise binary: in0 (A), in1 (B) -> add / sub / mul -> Dest -> pack -> out0, using only the id-free
// 2.0 compute API (explicit Dest sections; the ops take Dest inside them).
//
// Compile-time args: num_tiles, tiles_per_section (every section is one *_block<T, 1>; no tails, so num_tiles must
// divide evenly), op (0 add, 1 sub, 2 mul; broadcast, tiles_per_section 1: 3 sub COL, 4 mul COL, 5 mul ROW,
// 6 unary ROW, 7 unary COL), mode: 0 = one run, 6 = client switch (copy of A for 3
// sections, binary 5, copy 3), 7 = two DFB sets (A op B -> out0 then B op A -> out1, num_tiles / 2 each).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/bcast.h"
#include "api/compute/experimental/2_0/eltwise_binary.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

constexpr std::uint32_t TPS = get_arg(args::tiles_per_section);
constexpr std::uint32_t OP = get_arg(args::op);
static_assert(OP < 3 || TPS == 1, "broadcast ops run one tile per section");

template <std::uint32_t NumTiles, typename A, typename B, typename Out, typename Dfb>
void binary_sections(Dfb& a, Dfb& b, Dfb& out) {
    static_assert(NumTiles % TPS == 0, "no tails: tiles_per_section must divide the tile count");
    auto init = [&] {
        if constexpr (OP == 0) {
            ex::add_init(a.template front<A>(), b.template front<B>());
        } else if constexpr (OP == 1) {
            ex::sub_init(a.template front<A>(), b.template front<B>());
        } else if constexpr (OP == 2) {
            ex::mul_init(a.template front<A>(), b.template front<B>());
        } else if constexpr (OP == 3) {
            ex::bcast_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(a.template front<A>(), b.template front<B>());
        } else if constexpr (OP == 4) {
            ex::bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(a.template front<A>(), b.template front<B>());
        } else if constexpr (OP == 5) {
            ex::bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(a.template front<A>(), b.template front<B>());
        } else if constexpr (OP == 6) {
            ex::unary_bcast_src_to_dest_init<BroadcastType::ROW>(a.template front<A>());
        } else {
            ex::unary_bcast_src_to_dest_init<BroadcastType::COL>(a.template front<A>());
        }
        ex::pack_init(out.template back<Out>());
    };
    init();
    for (std::uint32_t done = 0; done < NumTiles; done += TPS) {
        a.wait_front(TPS);
        b.wait_front(TPS);
        out.reserve_back(TPS);
        ckernel::tile_regs_acquire();
        if constexpr (OP == 3) {
            ex::sub_tiles_bcast<BroadcastType::COL>(a.template front<A>(), b.template front<B>(), 0, 0, 0);
        } else if constexpr (OP == 4) {
            ex::mul_tiles_bcast<BroadcastType::COL>(a.template front<A>(), b.template front<B>(), 0, 0, 0);
        } else if constexpr (OP == 5) {
            ex::mul_tiles_bcast<BroadcastType::ROW>(a.template front<A>(), b.template front<B>(), 0, 0, 0);
        } else if constexpr (OP == 6) {
            ex::unary_bcast_src_to_dest_tile<BroadcastType::ROW>(a.template front<A>(), 0);
        } else if constexpr (OP == 7) {
            ex::unary_bcast_src_to_dest_tile<BroadcastType::COL>(a.template front<A>(), 0);
        } else if constexpr (OP == 0) {
            ex::add_block<TPS, 1>(a.template front<A>(), b.template front<B>(), 0, 0);
        } else if constexpr (OP == 1) {
            ex::sub_block<TPS, 1>(a.template front<A>(), b.template front<B>(), 0, 0);
        } else {
            ex::mul_block<TPS, 1>(a.template front<A>(), b.template front<B>(), 0, 0);
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(TPS);
        a.pop_front(TPS);
        b.pop_front(TPS);
    }
}

template <std::uint32_t NumTiles, typename In, typename Out, typename Dfb>
void copy_sections(Dfb& in, Dfb& out) {
    static_assert(NumTiles % TPS == 0, "no tails: tiles_per_section must divide the tile count");
    ex::copy_src_to_dest_init(in.template front<In>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < NumTiles; done += TPS) {
        in.wait_front(TPS);
        out.reserve_back(TPS);
        ckernel::tile_regs_acquire();
        ex::copy_src_to_dest_block<TPS, 1>(in.template front<In>(), 0);
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(TPS);
        in.pop_front(TPS);
    }
}

void kernel_main() {
    using A = LLKOperandFrom<dfb::in0>;
    using B = LLKOperandFrom<dfb::in1>;
    using Out0 = LLKOperandFrom<dfb::out0>;
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr std::uint32_t mode = get_arg(args::mode);

    DataflowBuffer a(dfb::in0);
    DataflowBuffer b(dfb::in1);
    DataflowBuffer out0(dfb::out0);
    ckernel::compute_kernel_hw_startup(a.template front<A>(), b.template front<B>(), out0.template back<Out0>());

    if constexpr (mode == 6) {
        copy_sections<3 * TPS, A, Out0>(a, out0);
        binary_sections<5 * TPS, A, B, Out0>(a, b, out0);
        copy_sections<3 * TPS, A, Out0>(a, out0);
    } else if constexpr (mode == 7) {
#if NUM_OUT > 1
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer out1(dfb::out1);
        binary_sections<num_tiles / 2, A, B, Out0>(a, b, out0);
        binary_sections<num_tiles / 2, B, A, Out1>(b, a, out1);
#endif
    } else {
        binary_sections<num_tiles, A, B, Out0>(a, b, out0);
    }
}
