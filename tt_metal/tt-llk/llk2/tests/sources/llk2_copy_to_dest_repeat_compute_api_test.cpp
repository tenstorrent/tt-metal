// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: copies straight to Dest, one after another, behind an op that leaves the math thread's address
// modifiers stepping. Only Tensors, the Compute API (copy, pack, complete) and a Dest ring.
//
// A copy straight to Dest issues one ZEROACC per face on the math thread (llk_lib/copy.h), and the 16-row ZEROACC
// applies its address modifier (tt-isa-documentation, ZEROACC.md). The copies here follow a transposed copy, which
// goes through SrcA and leaves the modifier slot those ZEROACCs name as "Dest on by eight rows". If a copy straight
// to Dest relied on what ran before, the Dest row counter would move on by 128 rows per copy of four tiles, with no
// FPU or SFPU configure in between to reset it. Blackhole picks the Dest half a face ZEROACC works in from offset +
// index + that counter: from the fourth straight copy on (the second into section 0) every one of its ZEROACCs
// would land in section 1 and mark the rows there, which the ring zeroed at the last release, as written again.
//
//   round 0        transposed copy (through SrcA) into section 0, packed to output tiles [0, T). With FIRST_IS_MATMUL
//                  a matmul of src with itself instead: it leaves every modifier slot the FPU ops share moving the
//                  Dest row counter, also the one a straight copy's ZEROACCs name
//   rounds 1 .. 4  straight copy into sections 1, 0, 1, 0, each packed to output tiles [T, 2T)
//   last           section 1 as the ring hands it out, written by nobody, packed to output tiles [2T, 3T): zero
//
// complete<Math>(out) after every release makes math wait until the pack thread has zeroed the section, so the order
// of the release's ZEROACC and the next copy's is the same on every run.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/sync.h"

constexpr DataFormat src_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& src_spec = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, src_format>;
inline constexpr auto& out_spec = llk2::l1_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<TILES_Y * llk2::default_tile_dim, TILES_X * llk2::default_tile_dim, llk2::kernel_dest_format>;

constexpr llk2::copy_modifiers transposed {.transpose = true};
static_assert(llk2::resolved<llk2::copy_op, src_spec, src_spec, transposed>.plan.in0.to == llk2::Where::SrcA, "round 0 goes through SrcA");
static_assert(!FIRST_IS_MATMUL || TILES_Y == TILES_X, "the matmul of src with itself needs a square tile grid");
static_assert(
    llk2::resolved<llk2::copy_op, src_spec, src_spec, llk2::copy_modifiers {}>.plan.in0.to == llk2::Where::Dest, "the other rounds go straight to Dest");
static_assert(llk2::kernel_dest_sections == 2 && STRAIGHT_COPIES % 2 == 0, "the last straight copy goes to section 0, the unwritten tensor is section 1");

/// Round 0's op. A template, so that only the form the build asks for is instantiated (the matmul needs a square grid).
template <bool Matmul, class Src, class Acc>
inline __attribute__((always_inline)) auto first_op(const Src& src, const Acc& acc)
{
    if constexpr (Matmul)
    {
        return llk2::matmul(src, src, acc);
    }
    else
    {
        return llk2::copy<transposed>(src, acc);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    using llk2::Thread;
    using AccRing                 = llk2::DestRing<acc_spec>;
    constexpr std::uint32_t tiles = TILES_Y * TILES_X;
    const auto src                = llk2::Tensor<src_spec>(params.buffer_A[0]);

    const auto first     = first_op<FIRST_IS_MATMUL>(src, llk2::reserve_back<AccRing>());
    const auto out_first = llk2::pack(first, llk2::Tensor<out_spec>(params.buffer_Res[0]));
    llk2::release(first);
    llk2::complete<Thread::Math>(out_first);

    for (std::uint32_t round = 0; round < STRAIGHT_COPIES; round++)
    {
        const auto y   = llk2::copy(src, llk2::reserve_back<AccRing>());
        const auto out = llk2::pack(y, llk2::Tensor<out_spec>(params.buffer_Res[tiles]));
        llk2::release(y);
        llk2::complete<Thread::Math>(out);
    }

    const auto zero = llk2::reserve_back<AccRing>(); // a section as the ring hands it out: zero
    llk2::pack(zero, llk2::Tensor<out_spec>(params.buffer_Res[2 * tiles]));
    llk2::release(zero);
}
