// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 eltwise binary: in0 (A), in1 (B) -> add / sub / mul -> Dest -> pack -> out0, using only the id-free
// 2.0 compute API and the take-driven Dest chain.
//
// Compile-time args: num_tiles, tiles_per_section (full sections use *_block<T, 1>, a shorter tail the runtime
// binary_block), op (0 add, 1 sub, 2 mul), mode: 0 = one run, 6 = chain switch (unpack-to-dest copy of A for 3
// sections, binary 5, copy 3), 7 = two DFB sets (A op B -> out0 then B op A -> out1, num_tiles / 2 each).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/dest_sync.h"
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
constexpr EltwiseBinaryType TYPE =
    OP == 0 ? EltwiseBinaryType::ELWADD : (OP == 1 ? EltwiseBinaryType::ELWSUB : EltwiseBinaryType::ELWMUL);

template <typename A, typename B, typename Out, typename Dfb>
void binary_sections(Dfb& a, Dfb& b, Dfb& out, std::uint32_t num_tiles) {
    if constexpr (OP == 0) {
        ex::add_init(a.template front<A>(), b.template front<B>());
    } else if constexpr (OP == 1) {
        ex::sub_init(a.template front<A>(), b.template front<B>());
    } else {
        ex::mul_init(a.template front<A>(), b.template front<B>());
    }
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < num_tiles;) {
        const std::uint32_t n = num_tiles - done < TPS ? num_tiles - done : TPS;
        a.wait_front(n);
        b.wait_front(n);
        out.reserve_back(n);
        ex::dest_acquire();
        if (n == TPS) {
            if constexpr (OP == 0) {
                ex::add_block<TPS, 1>(a.template front<A>(), b.template front<B>(), 0, 0);
            } else if constexpr (OP == 1) {
                ex::sub_block<TPS, 1>(a.template front<A>(), b.template front<B>(), 0, 0);
            } else {
                ex::mul_block<TPS, 1>(a.template front<A>(), b.template front<B>(), 0, 0);
            }
            ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        } else {
            ex::binary_block<TYPE>(a.template front<A>(), b.template front<B>(), 0, 0, 0, n);
            ex::pack_block(out.template back<Out>(), 0, 0, n);
        }
        ex::dest_release();
        out.push_back(n);
        a.pop_front(n);
        b.pop_front(n);
        done += n;
    }
}

template <typename In, typename Out, typename Dfb>
void copy_sections(Dfb& in, Dfb& out, std::uint32_t num_tiles) {
    ex::copy_init<true>(in.template front<In>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < num_tiles; done += TPS) {
        in.wait_front(TPS);
        out.reserve_back(TPS);
        ex::dest_acquire();
        ex::copy_block<TPS, 1, true>(in.template front<In>(), 0);
        ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        ex::dest_release();
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
        copy_sections<A, Out0>(a, out0, 3 * TPS);
        binary_sections<A, B, Out0>(a, b, out0, 5 * TPS);
        copy_sections<A, Out0>(a, out0, 3 * TPS);
    } else if constexpr (mode == 7) {
#if NUM_OUT > 1
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer out1(dfb::out1);
        binary_sections<A, B, Out0>(a, b, out0, num_tiles / 2);
        binary_sections<B, A, Out1>(b, a, out1, num_tiles / 2);
#endif
    } else {
        binary_sections<A, B, Out0>(a, b, out0, num_tiles);
    }
}
