// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// CROSS-LANE MIGRATION of the sdpa_reduce_row semantic lift (lane FK,
// 2026-08-22).  Lift parent (byte-untouched): semantic/sdpa_reduce_row.hpp
// (lane EW).  Original (byte-untouched):
//   kernel_includes/.../sfpu/experimental/ckernel_sfpu_sdpa_reduce_row.h
//
// The lift's raw sfpi::subvec_shflshr1/subvec_shflror1 lane folds are
// re-spelled on the graduated typed cross-lane surface (sfpi_crosslane.h):
//   shr1 chain of K            -> sfpi::subvec_slideup<K>  (result[col] =
//                                 col >= K ? v[col-K] : 0 -- exactly the
//                                 composed zero-fill SHFLSHR1 chain; BH
//                                 lowering is that chain verbatim, and the
//                                 X4 pass's R2 slide frames see it)
//   trailing ror1              -> sfpi::subvec_rotr<1>     (R1 rotate frame)
// The fold values are IDENTICAL: slideup's zero fill is the hardware
// SHFLSHR1 zero fill the original relies on (for MAX the zero enters the
// sign-magnitude compare exactly as in the original; for SUM it adds zero).
// At flag OFF this body and the lift must be CRAQ bit-exact (gate); the
// only .text delta class is scheduling (the lift hand-interleaved the two
// accumulator chains; the surface frames emit each chain whole and the
// list scheduler owns the interleave).
//
// Everything else (typed dst_reg walk, TT_SETC16/setrwc/semaphore protocol,
// the prev_max/prev_sum tails, the intentional differences documented in
// the lift header) is inherited verbatim.

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
namespace semantic {
namespace crosslane {

template <PoolType pool_type>
sfpi_inline sfpi::vFloat sdpa_reduce_op(sfpi::vFloat acc, sfpi::vFloat x) {
    if constexpr (pool_type == PoolType::MAX) {
        // SFPSWAP mod1=1 (== p_sfpswap::ALL_ROWS_MAX), max kept -- same insn as
        // the original's reduce_lregs_instr MAX arm.
        return sfpi::max(acc, x);
    } else {
        // SFPADD acc*1 + x -- same insn as the original's SUM arm.
        return acc + x;
    }
}

// One 8-row x 32-col DEST tile block (at dst_reg index ``base = 8*block``)
// folded into the two running accumulators, preserving the original's
// accumulation order and pairing exactly:
//   accA over vector rows {0, 1, 4, 5} (addresses 0, 2, 8, 10),
//   accB over vector rows {2, 3, 6, 7} (addresses 4, 6, 12, 14).
// first_block=true reproduces the original's replay(start+4, 12) entry: the
// first two rows initialize the accumulators instead of folding into them.
template <PoolType pool_type, bool first_block = false>
sfpi_inline void sdpa_reduce_row_8x32_block(const uint32_t base, sfpi::vFloat& accA, sfpi::vFloat& accB) {
    using namespace sfpi;
    if constexpr (first_block) {
        accA = dst_reg[base + 0];
        accB = dst_reg[base + 2];
    } else {
        accA = sdpa_reduce_op<pool_type>(accA, dst_reg[base + 0]);
        accB = sdpa_reduce_op<pool_type>(accB, dst_reg[base + 2]);
    }
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[base + 1]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[base + 3]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[base + 4]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[base + 6]);
    accA = sdpa_reduce_op<pool_type>(accA, dst_reg[base + 5]);
    accB = sdpa_reduce_op<pool_type>(accB, dst_reg[base + 7]);
}

// Lane-fold epilogue: reduce the 8 lanes of every 8-lane subvector row down to
// its lane 0 (slide-4 / slide-2 / slide-1 tree), then rotate right by one so
// the row scalar lands where the original leaves it.  Canonical surface
// frames for the original's SFPSHFT2 fold sequence.
template <DataFormat format, PoolType pool_type>
sfpi_inline void sdpa_reduce_row_8x32_epilogue(sfpi::vFloat& accA, sfpi::vFloat& accB) {
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "Unsupported data format. Supported formats: Float16_b");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::SUM,
        "Unsupported pool type. Supported pool types: MAX, SUM");

    accA = sdpa_reduce_op<pool_type>(accA, subvec_slideup<4>(accA));
    accB = sdpa_reduce_op<pool_type>(accB, subvec_slideup<4>(accB));
    accA = sdpa_reduce_op<pool_type>(accA, subvec_slideup<2>(accA));
    accB = sdpa_reduce_op<pool_type>(accB, subvec_slideup<2>(accB));
    accA = sdpa_reduce_op<pool_type>(accA, subvec_slideup<1>(accA));
    accB = sdpa_reduce_op<pool_type>(accB, subvec_slideup<1>(accB));
    accA = subvec_rotr<1>(accA);
    accB = subvec_rotr<1>(accB);
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    DataFormat format,
    uint32_t block_width,
    bool skip_signalling = false,
    uint32_t signal_granularity = 1>
sfpi_inline void sdpa_reduce_row_8x32_core(uint src_index, sfpi::vFloat& accA, sfpi::vFloat& accB) {
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

    sdpa_reduce_row_8x32_block<pool_type, /*first_block=*/true>(0, accA, accB);

    if constexpr (block_width > 1) {
        for (uint32_t i = 0; i < block_width - 1; i++) {
            if constexpr (!skip_signalling) {
                if ((i + 1) % signal_granularity == 0) {
                    t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
                    t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU);
                }
            }
            sdpa_reduce_row_8x32_block<pool_type>((i + 1) * 8, accA, accB);
        }
    }
    sdpa_reduce_row_8x32_epilogue<format, pool_type>(accA, accB);
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false, uint32_t signal_granularity = 1>
inline void _calculate_sdpa_reduce_max_row_8x32_(uint src_index, uint dst_index, bool prev_max = false) {
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    vFloat accA, accB;
    sdpa_reduce_row_8x32_core<
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
        // NOTE: the original re-loads the previous max into LREG1/LREG3 here
        // ("restore so that prev remains cached"); nothing reads that trailing
        // register state, so the lift drops it (see the lift header).
    }
    dst_reg[0] = accA;
    dst_reg[2] = accB;
    if constexpr (!skip_signalling) {
        t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
    }
}

template <DataFormat format, uint32_t block_width, bool skip_signalling = false>
inline void _calculate_sdpa_reduce_sum_row_8x32_(uint src_index, uint dst_index, bool prev_sum = false) {
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "SFPU reduce max col only supports Float16_b format");

    vFloat accA, accB;
    sdpa_reduce_row_8x32_core<PoolType::SUM, ReduceDim::REDUCE_ROW, format, block_width, skip_signalling>(
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

}  // namespace crosslane
}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
