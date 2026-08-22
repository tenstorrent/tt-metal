// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SEMANTIC LIFT of blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/
// common/inc/sfpu/experimental/ckernel_sfpu_sdpa_reduce_row.h (hand-scheduled
// raw-TTI + replay-buffer original).  The original is byte-untouched.
//
// Typed-SFPI body: same per-lane math and DEST walk, no raw SFPU TTI and no
// hand replay buffers (replay re-derivation is the compiler's job, e.g.
// -mtt-tensix-optimize-replay-hoist).  Retained as-is because they are protocol
// /configuration, not SFPU math: TT_SETC16 dest-base config (no typed
// equivalent), lltt::setrwc (typed SETRWC wrapper), and the FPU->SFPU
// t6_semaphore_* handshake.
//
// Instruction-semantics bridges (see SEMANTIC-LIFT.md for the full argument):
//   * sfpi::dst_reg[i] <-> SFPLOAD/SFPSTORE address 2*i; default vFloat access
//     emits mod0 0, matching the original's TTI_SFPLOAD/TTI_SFPSTORE mod0 0.
//   * sfpi::max(a, b) emits SFPSWAP mod1 = 1 -- the exact instruction and mod
//     the original uses (p_sfpswap::ALL_ROWS_MAX == 1 == vec_min_max: VC = max,
//     VD = min per lane), so NaN/sign-magnitude compare semantics are identical
//     by construction.
//   * sfpi::subvec_shflror1/subvec_shflshr1 emit SFPSHFT2 mod1
//     SUBVEC_SHFLROR1/SUBVEC_SHFLSHR1 -- the original's cross-lane folds,
//     builtin-for-instruction.  CAVEAT: these sfpi wrappers have no other
//     in-tree users (lane-S4 finding); the emitted words are identical to the
//     original's, so any semantics question applies to both bodies equally.
//   * The original walks the DEST counter one 8x32 block per replayed tile via
//     an ADDR_MOD with dest.incr = 16 on the last load.  TTINCRWC's immediate
//     is only [-8, 7], so a typed `dst_reg += 8` cannot encode that walk; the
//     lift instead addresses block b at absolute indices dst_reg[8*b + k]
//     (block_width is a template constant, so every index folds to an
//     immediate).  Same DEST cells, no running counter.
//
// Intentional architectural-state difference (documented, not a math change):
// the original's prev_max path re-loads the previous max into LREG1/LREG3 after
// consuming it ("Restore so that prev remains cached").  Nothing in this file
// or its callers reads those registers afterwards; a typed body cannot (and
// should not) pin values into named LREGs, so the lift drops that trailing
// side effect.  If some caller relies on an undocumented cross-call LREG
// contract, it must use the original.

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_instr_params.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
namespace semantic {

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
// its lane 0 (shift-4 / shift-2 / shift-1 tree), then rotate right by one so
// the row scalar lands where the original leaves it.  Same SFPSHFT2 fold
// sequence and reduce ops as _sdpa_reduce_row_8x32_epilogue_.
template <DataFormat format, PoolType pool_type>
sfpi_inline void sdpa_reduce_row_8x32_epilogue(sfpi::vFloat& accA, sfpi::vFloat& accB) {
    using namespace sfpi;
    static_assert(format == DataFormat::Float16_b, "Unsupported data format. Supported formats: Float16_b");
    static_assert(
        pool_type == PoolType::MAX || pool_type == PoolType::SUM,
        "Unsupported pool type. Supported pool types: MAX, SUM");

    // Shift 4x
    vFloat tA = subvec_shflshr1(accA);
    vFloat tB = subvec_shflshr1(accB);
    tA = subvec_shflshr1(tA);
    tB = subvec_shflshr1(tB);
    tA = subvec_shflshr1(tA);
    tB = subvec_shflshr1(tB);
    tA = subvec_shflshr1(tA);
    tB = subvec_shflshr1(tB);
    accA = sdpa_reduce_op<pool_type>(accA, tA);
    accB = sdpa_reduce_op<pool_type>(accB, tB);
    // Shift 2x
    tA = subvec_shflshr1(accA);
    tB = subvec_shflshr1(accB);
    tA = subvec_shflshr1(tA);
    tB = subvec_shflshr1(tB);
    accA = sdpa_reduce_op<pool_type>(accA, tA);
    accB = sdpa_reduce_op<pool_type>(accB, tB);
    // Shift 1x
    tA = subvec_shflshr1(accA);
    tB = subvec_shflshr1(accB);
    accA = sdpa_reduce_op<pool_type>(accA, tA);
    accB = sdpa_reduce_op<pool_type>(accB, tB);
    accA = subvec_shflror1(accA);
    accB = subvec_shflror1(accB);
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
        // register state, so the lift drops it (see file header).
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

}  // namespace semantic
}  // namespace sfpu
}  // namespace ckernel
