// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 matmul: in0, in1 -> matmul (accumulating over kt in Dest) -> pack -> out0, using only the id-free
// 2.0 compute API (legacy tile_regs_* Dest sections; UNPACK ops borrow Dest from MATH inside them). The readers
// stream each section's operand tiles in the order the section consumes them, so every tile is used once and popped.
//
// Compile-time args: num_tiles (output tiles), kt, block, mode, transpose.
//   block = 0 (tile mode): tiles_per_section output tiles per section; output tile j of a section is
//     sum_k in0[j * kt + k] x in1[j * kt + k] (matmul_tiles), with each in1 tile transposed when transpose = 1.
//     tiles_per_section divides num_tiles (no tails, static_assert).
//   block = 1 (block mode): ct x rt output tiles per section, C = in0[rt, kt] x in1[kt, ct] (both row-major,
//     matmul_block<ct, rt, kt> for k < kt). ct * rt divides num_tiles (no tails, static_assert).
//   mode: 0 = one run, 6 = client switch (copy of C for 3 sections, tile-mode matmul 5, copy 3),
//     7 = two DFB sets (in0 x in1 -> out0 then in1 x in0 -> out1, num_tiles / 2 each, tile mode),
//     11 = copy + matmul in one section (copy, FPU, PACK).
//   C, the copy source of modes 6 and 11, is in0, or in2 when bound (NUM_IN > 2): a Float32 in2 at a 32-bit Dest
//     makes the copy unpack-to-dest, so mode 6 switches UNPACK -> PACK and FPU -> PACK uses on the same regions and
//     mode 11 runs UNPACK -> FPU -> PACK with 16-bit matmul operands.

#include <cstdint>
#include <type_traits>

#include "api/compute/common.h"
#include "api/compute/reg_api.h"
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
constexpr std::uint32_t TRANSPOSE = get_arg(args::transpose);

// The copy source C takes the direct (unpack-to-dest) path when it is Float32 (run at a 32-bit Dest only).
template <typename C>
constexpr bool DIRECT = C::descriptor.format == DataFormat::Float32;

template <typename C>
void copy_init(C c) {
    if constexpr (DIRECT<C>) {
        ex::copy_direct_to_dest_init(c);
    } else {
        ex::copy_src_to_dest_init(c);
    }
}

template <std::uint32_t N, typename C>
void copy_block(C c) {
    if constexpr (DIRECT<C>) {
        ex::copy_direct_to_dest_block<N, 1>(c, 0);
    } else {
        ex::copy_src_to_dest_block<N, 1>(c, 0);
    }
}

template <std::uint32_t N, typename I0, typename I1, typename Out, typename Dfb>
void matmul_tile_sections(Dfb& in0, Dfb& in1, Dfb& out) {
    static_assert(N % TPS == 0, "no tails: tiles_per_section must divide the tile count");
    ex::matmul_init(in0.template front<I0>(), in1.template front<I1>(), TRANSPOSE);
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < N; done += TPS) {
        in0.wait_front(TPS * KT);
        in1.wait_front(TPS * KT);
        out.reserve_back(TPS);
        ckernel::tile_regs_acquire();
        for (std::uint32_t j = 0; j < TPS; ++j) {
            for (std::uint32_t k = 0; k < KT; ++k) {
                ex::matmul_tiles(in0.template front<I0>(), in1.template front<I1>(), j * KT + k, j * KT + k, j);
            }
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(TPS);
        in0.pop_front(TPS * KT);
        in1.pop_front(TPS * KT);
    }
}

template <std::uint32_t N, typename I0, typename I1, typename Out, typename Dfb>
void matmul_block_sections(Dfb& in0, Dfb& in1, Dfb& out) {
    static_assert(N % (CT * RT) == 0, "no tails: the ct x rt block must divide the tile count");
    ex::matmul_block_init<CT, RT, KT>(in0.template front<I0>(), in1.template front<I1>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < N; done += CT * RT) {
        in0.wait_front(RT * KT);
        in1.wait_front(KT * CT);
        out.reserve_back(CT * RT);
        ckernel::tile_regs_acquire();
        for (std::uint32_t k = 0; k < KT; ++k) {
            ex::matmul_block<CT, RT, KT>(in0.template front<I0>(), in1.template front<I1>(), k, k * CT, 0);
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<CT, RT>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(CT * RT);
        in0.pop_front(RT * KT);
        in1.pop_front(KT * CT);
    }
}

template <std::uint32_t N, typename In, typename Out, typename Dfb>
void copy_sections(Dfb& in, Dfb& out) {
    static_assert(N % TPS == 0, "no tails: tiles_per_section must divide the tile count");
    copy_init(in.template front<In>());
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t done = 0; done < N; done += TPS) {
        in.wait_front(TPS);
        out.reserve_back(TPS);
        ckernel::tile_regs_acquire();
        copy_block<TPS>(in.template front<In>());
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<TPS, 1>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(TPS);
        in.pop_front(TPS);
    }
}

// Copy then matmul on top, one tile per section. C == I0 (same DFB): Dest = in0[2t] + in0[2t + 1] x in1[t], 16-bit, so
// both are FPU. C = Float32 in2 at a 32-bit Dest: Dest = in2[t] + in0[t] x in1[t], UNPACK -> FPU -> PACK.
template <std::uint32_t N, typename C, typename I0, typename I1, typename Out, typename Dfb>
void copy_matmul_sections(Dfb& cin, Dfb& in0, Dfb& in1, Dfb& out) {
    constexpr bool same = std::is_same_v<C, I0>;
    constexpr std::uint32_t a = same ? 1 : 0;  // in0 index of the matmul operand
    ex::pack_init(out.template back<Out>());
    for (std::uint32_t t = 0; t < N; ++t) {
        in0.wait_front(1 + a);
        if constexpr (!same) {
            cin.wait_front(1);
        }
        in1.wait_front(1);
        out.reserve_back(1);
        ckernel::tile_regs_acquire();
        copy_init(cin.template front<C>());
        copy_block<1>(cin.template front<C>());
        ex::matmul_init(in0.template front<I0>(), in1.template front<I1>());
        ex::matmul_tiles(in0.template front<I0>(), in1.template front<I1>(), a, 0, 0);
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<1, 1>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(1);
        in0.pop_front(1 + a);
        if constexpr (!same) {
            cin.pop_front(1);
        }
        in1.pop_front(1);
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

#if NUM_IN > 2
    using C = LLKOperandFrom<dfb::in2>;
    DataflowBuffer cin(dfb::in2);
#else
    using C = I0;
    auto& cin = in0;
#endif
    if constexpr (mode == 6) {
        copy_sections<3 * TPS, C, Out0>(cin, out0);
        matmul_tile_sections<5 * TPS, I0, I1, Out0>(in0, in1, out0);
        copy_sections<3 * TPS, C, Out0>(cin, out0);
    } else if constexpr (mode == 11) {
        copy_matmul_sections<num_tiles, C, I0, I1, Out0>(cin, in0, in1, out0);
    } else if constexpr (mode == 7) {
#if NUM_OUT > 1
        using Out1 = LLKOperandFrom<dfb::out1>;
        DataflowBuffer out1(dfb::out1);
        matmul_tile_sections<num_tiles / 2, I0, I1, Out0>(in0, in1, out0);
        matmul_tile_sections<num_tiles / 2, I1, I0, Out1>(in1, in0, out1);
#endif
    } else if constexpr (block) {
        matmul_block_sections<num_tiles, I0, I1, Out0>(in0, in1, out0);
    } else {
        matmul_tile_sections<num_tiles, I0, I1, Out0>(in0, in1, out0);
    }
}
