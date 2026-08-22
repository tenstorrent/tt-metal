// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// WALK variant of the lane-EW semantic lift in sdpa_reduce_row.hpp (lane FI,
// envelope-attack).  Same per-lane math, same DEST cells, same accumulation
// order; ONE delivery-shape change:
//
//   The lift addresses block b at absolute indices dst_reg[8*b + k] (every
//   SFPLOAD gets a distinct immediate), which makes the unrolled block bodies
//   ENCODING-VARIANT — no replay window can capture them (the compiler's
//   replay passes correctly refuse variant encodings), so the whole 4-block
//   phase is straight-pushed: ~64 issued words/tile vs the original's 4 replay
//   launches of a buffer recorded once at init.  At the KERNEL (e2e) metric
//   this costs +3..4 cycles/tile (headline-blaze-multitile-20260822: max
//   +1.61%, sum +2.81% at t32); at the BLAZE_BODY diag it reads as +16..21%
//   because the original's fire-and-forget launches drain past its zone end.
//
//   This variant walks DEST with bounded typed increments instead: every block
//   body reads dst_reg[0..7] and then advances the base (dst_reg += 3/3/2 —
//   TTINCRWC's immediate is [-8,7] in address units, one index = 2 units, so
//   +8 indexes = +16 units takes three encodable steps).  The block bodies
//   become ADDRESS-INVARIANT, and the block loop is kept rolled
//   (#pragma GCC unroll 1) so the compiler's replay machinery sees a counted
//   loop with an invariant tensix body — the shape it CAN legally capture.
//   No lltt::record / lltt::replay appears here: replay re-derivation stays
//   the compiler's job (the lift charter), this file only stops hiding the
//   walk from it.
//
// The math tracks sdpa_reduce_row.hpp exactly (same sdpa_reduce_op /
// epilogue); see that file's header for the instruction-semantics bridges.

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sdpa_reduce_row.hpp"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

// Advance the DEST window by 8 vector rows (+16 address units) in TTINCRWC-
// encodable steps.  3 + 3 + 2 indexes = 6 + 6 + 4 address units, each <= 7.
sfpi_inline void sdpa_reduce_row_advance_8() {
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

// One 8-row x 32-col DEST tile block at the CURRENT dst_reg base, folded into
// the two running accumulators; same order/pairing as
// sdpa_reduce_row_8x32_block, then the window advances to the next block.
template <PoolType pool_type, bool first_block = false>
sfpi_inline void sdpa_reduce_row_8x32_block_walk(sfpi::vFloat& accA, sfpi::vFloat& accB) {
    using namespace sfpi;
    if constexpr (first_block) {
        accA = dst_reg[0];
        accB = dst_reg[2];
    } else {
        accA = sdpa_reduce_op<pool_type>(accA, dst_reg[0]);
        accB = sdpa_reduce_op<pool_type>(accB, dst_reg[2]);
    }
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[1]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[3]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[4]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[6]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[5]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[7]);
    sdpa_reduce_row_advance_8();
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    uint32_t block_width,
    bool skip_signalling = false,
    uint32_t signal_granularity = 1>
sfpi_inline void sdpa_reduce_row_8x32_core_walk(uint src_index, sfpi::vFloat& accA, sfpi::vFloat& accB) {
    static_assert(reduce_dim == ReduceDim::REDUCE_ROW, "Only row reduction (REDUCE_ROW) is currently supported");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::SUM,
        "Unsupported pool type. Supported pool types: MAX, SUM");
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");
    static_assert(block_width % signal_granularity == 0, "block_width must be divisible by signal_granularity");

    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, src_index + get_dest_buffer_base());
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();

    if constexpr (!skip_signalling) {
        t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
    }

    sdpa_reduce_row_8x32_block_walk<pool_type, /*first_block=*/true>(accA, accB);

    if constexpr (block_width > 1) {
        // Unroll the block loop: the bodies are now address-invariant (the
        // walk is carried by the trailing increments), so the unrolled copies
        // are ENCODING-IDENTICAL repeats — the shape the replay pass captures
        // (record the first copy with exec, launch the rest).  A rolled loop
        // does NOT form: the pass derives windows from intra-block repetition,
        // not from counted loops (lane-FI probe, laneFI-evidence-20260822).
#pragma GCC unroll 3
        for (uint32_t i = 0; i < block_width - 1; i++) {
            if constexpr (!skip_signalling) {
                if ((i + 1) % signal_granularity == 0) {
                    t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
                    t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
                }
            }
            sdpa_reduce_row_8x32_block_walk<pool_type>(accA, accB);
        }
    }
    sdpa_reduce_row_8x32_epilogue<format, pool_type>(accA, accB);
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false, uint32_t signal_granularity = 1>
inline void _calculate_sdpa_reduce_max_row_8x32_walk_(uint src_index, uint dst_index, bool prev_max = false) {
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    vFloat accA, accB;
    sdpa_reduce_row_8x32_core_walk<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        format,
        block_width,
        skip_signalling,
        signal_granularity>(src_index, accA, accB);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    if (prev_max) {
        accA = sdpa_reduce_op<PoolType::MAX>(accA, dst_reg[0]);
        accB = sdpa_reduce_op<PoolType::MAX>(accB, dst_reg[2]);
        // See sdpa_reduce_row.hpp: the original's trailing LREG1/LREG3
        // "prev remains cached" re-loads are dead state, dropped by the lift.
    }
    dst_reg[0] = accA;
    dst_reg[2] = accB;
    if constexpr (!skip_signalling) {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false>
inline void _calculate_sdpa_reduce_sum_row_8x32_walk_(uint src_index, uint dst_index, bool prev_sum = false) {
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    vFloat accA, accB;
    sdpa_reduce_row_8x32_core_walk<PoolType::SUM, ReduceDim::REDUCE_ROW, format, block_width, skip_signalling>(
        src_index, accA, accB);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    lltt::setrwc<p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D>();
    if (prev_sum) {
        accA = sdpa_reduce_op<PoolType::SUM>(accA, dst_reg[0]);
        accB = sdpa_reduce_op<PoolType::SUM>(accB, dst_reg[2]);
    }
    dst_reg[0] = accA;
    dst_reg[2] = accB;
    if constexpr (!skip_signalling) {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
