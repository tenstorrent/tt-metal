// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lane IE UNIFORM-BLOCK semantic twin of the lane-EW sdpa_reduce_row lift
// (helpers/include/blaze_vendored/blaze/kernels/sfpu/semantic/
// sdpa_reduce_row.hpp).  Test-side twin (R7: LLK trees and blaze_vendored
// stay pristine; this file lives outside the vendored root and is harness
// property).  Same per-lane math, same DEST cells, same per-accumulator fold
// order; NO raw SFPU TTI, NO hand replay buffers — replay/delivery
// re-derivation stays the compiler's job (the lane-FI kernel-envelope
// charter: express the ALGORITHM in a shape the compiler can reward).
//
// Why the lift straight-pushes 64 words/tile (laneHW row A9/A10, laneFI §2):
// the lift addresses block b at absolute indices dst_reg[8*b + k], so the
// four unrolled block bodies are ENCODING-VARIANT and the replay former
// correctly refuses them.  The lane-FI walk variant (sdpa_reduce_row_walk.hpp)
// made blocks address-invariant but kept TWO non-uniformities, both visible
// in the pin-38/ON-36 census (laneIE probes):
//   1. the FIRST block initializes the accumulators (pure loads) while blocks
//      2..4 fold — so the former can only capture the 12-word common tail and
//      the 2-word heads stay raw per block;
//   2. the dst_reg += 3/3/2 block-walk (TTINCRWC imm is [-8,7] address units)
//      trails the captured window — 12 raw TTINCRWC/tile.
//
// This twin removes both:
//   * SEEDED ACCUMULATORS: accA/accB start at the reduction identity
//     (+0.0f for SUM, -inf for MAX), so ALL FOUR blocks are pure fold blocks
//     — encoding-identical including their heads.  Numerically exact:
//     max(-inf, x) == x for every non-NaN x and propagates NaN exactly as the
//     original's acc-init does under sign-magnitude SFPSWAP max; 0.0f + x is
//     bit-exact for every x except x == -0.0 (which the +0.0 seed rounds to
//     +0.0; the vehicle's stimuli are uniform(0.1, 2.0)).  Device-golden
//     authority; documented here.
//   * ENCODABLE INTERLEAVED WALK: the block reads {0,2, 1,3, 4,6, 5,7} become
//     four uniform pair-steps, each reading dst_reg[0] -> accA and
//     dst_reg[2] -> accB at the current window then advancing the window by
//     +1, +3, +1, +3 rows (2/6/2/6 address units — every step a single
//     encodable TTINCRWC, absorbable by -mtt-tensix-optimize-dst-autoincr).
//     Absolute addresses and per-accumulator fold order are IDENTICAL to the
//     lift's: accA folds rows 0,1,4,5 and accB rows 2,3,6,7 of every block,
//     in that order — bit-exact SUM chains, identical SWAP chains.
//
// The tile phase is then 16 identical 5-word pair-steps (an alternating
// +1/+3 walk => a 10-word repeating unit), the shape the replay former
// captures from unrolled encoding-identical copies (laneFI probe fact) and
// the record-hoist-loop pass (laneFW) can hoist out of the runtime tile loop.
//
// WALK8 variant (uniform8): same seeded-uniform blocks, lane-FI's original
// +3/+3/+2 end-of-block walk kept — isolates the head-uniformity effect from
// the walk-encoding effect (probe pair; the census picks the winner).
//
// Epilogue and result-store tail are the lift's, unchanged (shared helpers
// included from the vendored lift header).

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
// Shared helpers (sdpa_reduce_op, sdpa_reduce_row_8x32_epilogue) from the
// vendored lift; resolved via -Ihelpers/include/blaze_vendored.
#include "blaze/kernels/sfpu/semantic/sdpa_reduce_row.hpp"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{
namespace semantic
{

// Materialized per calculate call (a vFloat cannot cross the SFPU_UNARY_CALL
// boundary: "cannot write SFPU object to memory" — laneIE probe on38-*-e).
template <PoolType pool_type>
sfpi_inline sfpi::vFloat sdpa_reduce_identity()
{
    if constexpr (pool_type == PoolType::MAX)
    {
        // -inf: the sign-magnitude SFPSWAP max identity (0xFF800000, bf16-
        // encodable as 0xFF80 so the loadi is a single word).
        return sfpi::vFloat(-__builtin_inff());
    }
    else
    {
        // -0.0: the EXACT FP additive identity under round-to-nearest for
        // EVERY x (including +-0.0 and NaN payload propagation), so the
        // seeded chain is bit-identical to the lift's acc-init chain for all
        // inputs.  The bit pattern is laundered through a volatile local
        // because the compiler constant-tracks every direct spelling (a 0.0f
        // literal and vConst0 fold to nothing; a reinterpret'd 0x80000000
        // vInt still constant-propagates into an SFPADDI) — each of which
        // re-introduces the non-uniform init head the seed exists to remove
        // (laneIE probes on38-uni8{,-b,-c} sum).  The value is a compile-time
        // constant in every execution; only its VISIBILITY to the folder is
        // suppressed.
        volatile std::uint32_t seed_bits = 0x80000000u;
        return sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(static_cast<int>(seed_bits)));
    }
}

// One uniform pair-step: fold the vector row at the current window base into
// accA and the row two below into accB, then advance the window by ADV rows.
// ADV in {1, 3}: 2 or 6 address units — a single encodable TTINCRWC.
template <PoolType pool_type, int ADV>
sfpi_inline void sdpa_reduce_row_pair_step(sfpi::vFloat& accA, sfpi::vFloat& accB)
{
    using namespace sfpi;
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[0]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[2]);
    dst_reg += ADV;
}

// One 8-row x 32-col block as four uniform pair-steps.  Window walk within
// the block: base+0 -> +1 -> +4 -> +5 -> next block's base (+8 total), so the
// reads hit rows {0,2},{1,3},{4,6},{5,7} — the lift's exact read order and
// accumulator pairing.
template <PoolType pool_type>
sfpi_inline void sdpa_reduce_row_8x32_block_uniform(sfpi::vFloat& accA, sfpi::vFloat& accB)
{
    sdpa_reduce_row_pair_step<pool_type, 1>(accA, accB);
    sdpa_reduce_row_pair_step<pool_type, 3>(accA, accB);
    sdpa_reduce_row_pair_step<pool_type, 1>(accA, accB);
    sdpa_reduce_row_pair_step<pool_type, 3>(accA, accB);
}

// HALF-BLOCK (shape 2): four rows per uniform unit — reads {0,2},{1,3} at
// the current window, then advances +3 +1 (6 and 2 address units, both
// encodable) to the next half-block.  The 10-word unit repeats 2*block_width
// times per tile, ALL copies encoding-identical, and 10 words fit the replay
// buffer's free half (slots 0..15; the envelope datacopy record owns 16..31)
// with room left for an epilogue window — the constraint that forces the
// full-block capture (shape 1, 16 words) to evict the epilogue record.
// Per-accumulator fold order is again the lift's exactly (accA rows
// 0,1,4,5,...; accB rows 2,3,6,7,... of the tile).
template <PoolType pool_type>
sfpi_inline void sdpa_reduce_row_4x32_halfblock(sfpi::vFloat& accA, sfpi::vFloat& accB)
{
    using namespace sfpi;
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[0]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[2]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[1]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[3]);
    dst_reg += 3;
    dst_reg += 1;
}

// SEQ block (shape 3): the tile's 32 rows read in PURE ascending order with
// a unit-stride walk (load dst_reg[0]; dst_reg += 1) — the canonical carrier
// shape of -mtt-tensix-optimize-dst-autoincr (probe: does the pass fold the
// TTINCRWC into the load's ADDR_MOD the way the hand kernel carries its
// walk?).  Routing rows r%4<2 -> accA, else accB preserves each
// accumulator's fold order exactly (accA gets 0,1,4,5,...; accB 2,3,6,7,...
// — only the A/B interleaving changes, and the chains are independent).
template <PoolType pool_type>
sfpi_inline void sdpa_reduce_row_seq_quad(sfpi::vFloat& accA, sfpi::vFloat& accB)
{
    using namespace sfpi;
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[0]);
    dst_reg += 1;
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[0]);
    dst_reg += 1;
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[0]);
    dst_reg += 1;
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[0]);
    dst_reg += 1;
}

// WALK8 block: the lift's read order at fixed offsets within the window,
// then lane-FI's +3/+3/+2 end-of-block walk (all steps encodable).
template <PoolType pool_type>
sfpi_inline void sdpa_reduce_row_8x32_block_uniform8(sfpi::vFloat& accA, sfpi::vFloat& accB)
{
    using namespace sfpi;
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[0]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[2]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[1]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[3]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[4]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[6]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[5]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[7]);
    dst_reg += 3;
    dst_reg += 3;
    dst_reg += 2;
}

// SHAPE: 0 = pair-step blocks (alternating +1/+3 walk), 1 = walk8 blocks
// (lane-FI +3/+3/+2 end-of-block walk), 2 = half-blocks (10-word units).
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    std::uint32_t block_width,
    int shape,
    bool skip_signalling             = false,
    std::uint32_t signal_granularity = 1>
sfpi_inline void sdpa_reduce_row_8x32_core_uniform(std::uint32_t src_index, sfpi::vFloat& accA, sfpi::vFloat& accB)
{
    static_assert(reduce_dim == ReduceDim::REDUCE_ROW, "Only row reduction (REDUCE_ROW) is currently supported");
    static_assert(pool_type == PoolType::MAX || pool_type == PoolType::SUM, "Unsupported pool type. Supported pool types: MAX, SUM");
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");
    static_assert(block_width % signal_granularity == 0, "block_width must be divisible by signal_granularity");

    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index + get_dest_buffer_base());
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();

    if constexpr (!skip_signalling)
    {
        t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
    }

    // Two independent materializations on purpose: a shared seed register
    // copy-propagates into block 0's first folds and breaks the encoding
    // uniformity the seed exists to create (laneIE probe on38-*-f).
    accA = sdpa_reduce_identity<pool_type>();
    accB = sdpa_reduce_identity<pool_type>();

    if constexpr (shape == 2)
    {
        static_assert(skip_signalling, "half-block shape carries no per-block signalling");
#pragma GCC unroll 8
        for (std::uint32_t i = 0; i < 2 * block_width; i++)
        {
            sdpa_reduce_row_4x32_halfblock<pool_type>(accA, accB);
        }
    }
    else if constexpr (shape == 3)
    {
        static_assert(skip_signalling, "seq shape carries no per-block signalling");
#pragma GCC unroll 8
        for (std::uint32_t i = 0; i < 2 * block_width; i++)
        {
            sdpa_reduce_row_seq_quad<pool_type>(accA, accB);
        }
    }
    else
    {
#pragma GCC unroll 4
        for (std::uint32_t i = 0; i < block_width; i++)
        {
            if constexpr (!skip_signalling)
            {
                if (i > 0 && i % signal_granularity == 0)
                {
                    t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
                    t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
                }
            }
            if constexpr (shape == 1)
            {
                sdpa_reduce_row_8x32_block_uniform8<pool_type>(accA, accB);
            }
            else
            {
                sdpa_reduce_row_8x32_block_uniform<pool_type>(accA, accB);
            }
        }
    }
    sdpa_reduce_row_8x32_epilogue<format, pool_type>(accA, accB);
}

template <DataFormat format, std::uint32_t block_width, int shape, bool skip_signalling = false, std::uint32_t signal_granularity = 1>
inline void _calculate_sdpa_reduce_max_row_8x32_uniform_(std::uint32_t src_index, std::uint32_t dst_index, bool prev_max = false)
{
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    vFloat accA, accB;
    sdpa_reduce_row_8x32_core_uniform<PoolType::MAX, ReduceDim::REDUCE_ROW, format, block_width, shape, skip_signalling, signal_granularity>(
        src_index, accA, accB);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    if (prev_max)
    {
        accA = sdpa_reduce_op<PoolType::MAX>(accA, dst_reg[0]);
        accB = sdpa_reduce_op<PoolType::MAX>(accB, dst_reg[2]);
        // See sdpa_reduce_row.hpp: the original's trailing LREG1/LREG3
        // "prev remains cached" re-loads are dead state, dropped by the lift.
    }
    dst_reg[0] = accA;
    dst_reg[2] = accB;
    if constexpr (!skip_signalling)
    {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

template <DataFormat format, std::uint32_t block_width, int shape, bool skip_signalling = false>
inline void _calculate_sdpa_reduce_sum_row_8x32_uniform_(std::uint32_t src_index, std::uint32_t dst_index, bool prev_sum = false)
{
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    vFloat accA, accB;
    sdpa_reduce_row_8x32_core_uniform<PoolType::SUM, ReduceDim::REDUCE_ROW, format, block_width, shape, skip_signalling>(src_index, accA, accB);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    if (prev_sum)
    {
        accA = sdpa_reduce_op<PoolType::SUM>(accA, dst_reg[0]);
        accB = sdpa_reduce_op<PoolType::SUM>(accB, dst_reg[2]);
    }
    dst_reg[0] = accA;
    dst_reg[2] = accB;
    if constexpr (!skip_signalling)
    {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

} // namespace semantic
} // namespace sfpu
} // namespace ckernel
