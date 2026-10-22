// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 matmul: in0, in1 -> matmul (accumulating over kt in Dest) -> pack -> out0, using only the id-free
// 2.0 compute API and the take-driven Dest chain. The readers stream each section's operand tiles in the order the
// section consumes them, so every tile is used once and popped.
//
// Compile-time args: num_tiles (output tiles), kt, block, mode.
//   block = 0 (tile mode): tiles_per_section output tiles per section; output tile j of a section is
//     sum_k in0[j * kt + k] x in1[j * kt + k] (matmul_tile). A shorter tail section re-inits first.
//   block = 1 (block mode): ct x rt output tiles per section, C = in0[rt, kt] x in1[kt, ct] (both row-major,
//     matmul_block<ct, rt> for k < kt). A tail of num_tiles % (ct * rt) tiles (rt must be 1) re-inits with the
//     runtime shape and uses the runtime matmul_block.
//   mode: 0 = one run, 6 = chain switch (unpack-to-dest copy of in0 for 3 sections, tile-mode matmul 5, copy 3),
//     7 = two DFB sets (in0 x in1 -> out0 then in1 x in0 -> out1, num_tiles / 2 each, tile mode).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/dest_sync.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/matmul.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

constexpr std::uint32_t TPS = get_arg(args::tiles_per_section);
constexpr std::uint32_t KT = get_arg(args::kt);
constexpr std::uint32_t CT = get_arg(args::ct);
constexpr std::uint32_t RT = get_arg(args::rt);

template <typename I0, typename I1, typename Out, typename Dfb>
void matmul_tile_sections(Dfb& in0, Dfb& in1, Dfb& out, std::uint32_t num_tiles) {
    ex::matmul_init<1, 1>(in0.template front<I0>(), in1.template front<I1>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < num_tiles;) {
        const std::uint32_t n = num_tiles - done < TPS ? num_tiles - done : TPS;
        if (n != TPS) {  // tail: block ops take no tails, so re-init before the shorter block
            ex::matmul_init<1, 1>(in0.template front<I0>(), in1.template front<I1>());
            ex::pack_init(out.template back<Out>());
        }
        in0.wait_front(n * KT);
        in1.wait_front(n * KT);
        out.reserve_back(n);
        ex::dest_acquire();
        for (std::uint32_t j = 0; j < n; ++j) {
            for (std::uint32_t k = 0; k < KT; ++k) {
                ex::matmul_tile(in0.template front<I0>(), in1.template front<I1>(), j * KT + k, j * KT + k, j);
            }
        }
        ex::pack_block(out.template back<Out>(), 0, 0, n);
        ex::dest_release();
        out.push_back(n);
        in0.pop_front(n * KT);
        in1.pop_front(n * KT);
        done += n;
    }
}

template <typename I0, typename I1, typename Out, typename Dfb>
void matmul_block_sections(Dfb& in0, Dfb& in1, Dfb& out, std::uint32_t num_tiles) {
    ex::matmul_init<CT, RT>(in0.template front<I0>(), in1.template front<I1>(), KT);
    ex::pack_init(out.template back<Out>());
    std::uint32_t done = 0;
    for (; done + CT * RT <= num_tiles; done += CT * RT) {
        in0.wait_front(RT * KT);
        in1.wait_front(KT * CT);
        out.reserve_back(CT * RT);
        ex::dest_acquire();
        for (std::uint32_t k = 0; k < KT; ++k) {
            ex::matmul_block<CT, RT>(in0.template front<I0>(), in1.template front<I1>(), k, k * CT, KT);
        }
        ex::pack_block<CT, RT>(out.template back<Out>(), 0);
        ex::dest_release();
        out.push_back(CT * RT);
        in0.pop_front(RT * KT);
        in1.pop_front(KT * CT);
    }
    if (done < num_tiles) {  // tail: a 1 x ct_tail block, runtime shape
        const std::uint32_t ct = num_tiles - done;
        ex::matmul_init(in0.template front<I0>(), in1.template front<I1>(), ct, 1, KT);
        ex::pack_init(out.template back<Out>());
        in0.wait_front(KT);
        in1.wait_front(KT * ct);
        out.reserve_back(ct);
        ex::dest_acquire();
        for (std::uint32_t k = 0; k < KT; ++k) {
            ex::matmul_block(in0.template front<I0>(), in1.template front<I1>(), k, k * ct, ct, 1, KT);
        }
        ex::pack_block(out.template back<Out>(), 0, 0, ct);
        ex::dest_release();
        out.push_back(ct);
        in0.pop_front(KT);
        in1.pop_front(KT * ct);
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
    using I0 = LLKOperandFrom<dfb::in0>;
    using I1 = LLKOperandFrom<dfb::in1>;
    using Out0 = LLKOperandFrom<dfb::out0>;
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr bool block = get_arg(args::block) != 0;
    constexpr std::uint32_t mode = get_arg(args::mode);

    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out0(dfb::out0);
    // matmul: in0 -> SrcB, in1 -> SrcA.
    ckernel::compute_kernel_hw_startup(in1.template front<I1>(), in0.template front<I0>(), out0.template back<Out0>());

    if constexpr (mode == 6) {
        copy_sections<I0, Out0>(in0, out0, 3 * TPS);
        matmul_tile_sections<I0, I1, Out0>(in0, in1, out0, 5 * TPS);
        copy_sections<I0, Out0>(in0, out0, 3 * TPS);
    } else if constexpr (mode == 7) {
#if NUM_OUT > 1
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer out1(dfb::out1);
        matmul_tile_sections<I0, I1, Out0>(in0, in1, out0, num_tiles / 2);
        matmul_tile_sections<I1, I0, Out1>(in1, in0, out1, num_tiles / 2);
#endif
    } else if constexpr (block) {
        matmul_block_sections<I0, I1, Out0>(in0, in1, out0, num_tiles);
    } else {
        matmul_tile_sections<I0, I1, Out0>(in0, in1, out0, num_tiles);
    }
}
