// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// imm12 bit 11 = 1: SFPSETCC interprets src_c as two's-complement INT32, not FP32/SMAG32
constexpr std::uint32_t SFPSETCC_INT32_SIGNBIT = 0x800;

// 9-instruction recorded body: 2 loads + SFPSWAP + SFPNOP + 2 SFPSETCC + SFPSWAP + SFPENCC + SFPSTORE.
constexpr std::uint32_t BINARY_MAX_MIN_REPLAY_LEN = 9;

// Int32 uses sfpmem::INT32; all float and MX formats use sfpmem::DEFAULT.
// DEFAULT lets SFPLOAD resolve the format at runtime via ALU_ACC_CTRL_SFPU_Fp32_enabled
// and the SrcB format register, which the unpacker already programs from formats.math.
template <DataFormat FMT>
inline constexpr std::uint32_t _binary_max_min_sfpmem_mode_() {
    return (FMT == DataFormat::Int32) ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::DEFAULT;
}

// Programs ADDR_MOD_6 with dest.incr=2 so the SFPSTORE in the replayed body
// auto-advances the dest counter by one SFP-row pair per iteration. Quasar's
// shared SFPU init only programs ADDR_MOD_7 (incr=0); this is additive.
inline void _init_binary_max_min_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

// Unified element-wise max/min over two Dest tile regions.
//
// Records the 9-instruction body once into replay slots 0..8; each REPLAY re-executes
// it verbatim. ADDR_MOD_6's dest.incr=2 on SFPSTORE (programmed by _init_binary_max_min_)
// advances the dest counter by one SFP-row pair after each replay so the per-row offsets
// stay constant in the recorded instructions and the replay buffer can re-issue them
// unchanged across iterations.
//
// FMT controls the SFPLOAD/SFPSTORE sfpmem mode:
//   DataFormat::Int32                                         → sfpmem::INT32
//   Float16 / Float16_b / Float32 / Tf32 / MxFp8R / MxFp8P  → sfpmem::DEFAULT
//
// SFPSWAP_VEC_MIN_MAX on Quasar uses an unsigned 32-bit compare on the LReg bits
// instead of the documented sign-magnitude (SignMagIsSmaller) compare. For (neg, neg)
// pairs that inverts the ordering for both FP32 sign-magnitude and INT32 sign-magnitude.
// The CC-guarded correction swap (SFPSETCC-on-LT0 for both operands, then SFPSWAP) fixes
// those rows. The correction is a no-op for unsigned-origin lanes (bit 31 always 0).
//
// @tparam FMT           Math-side DataFormat.
// @tparam IS_MAX_OP     true → store max(in0, in1); false → store min.
// @tparam ITERATIONS    Number of SFP-row pairs per face (8 for a 32×16 face).
// @param dst_index_in0  Dest tile index of input 0 (in tile units, relative to DST_INDEX).
// @param dst_index_in1  Dest tile index of input 1 (in tile units, relative to DST_INDEX).
// @param dst_index_out  Dest tile index where the result is written (in tile units, relative to DST_INDEX).
template <DataFormat FMT, bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 ||
            FMT == DataFormat::Tf32 || FMT == DataFormat::MxFp8R || FMT == DataFormat::MxFp8P ||
            FMT == DataFormat::Int32,
        "Unsupported DataFormat for calculate_binary_max_min().");

    constexpr std::uint32_t SFPMEM_MODE = _binary_max_min_sfpmem_mode_<FMT>();

    // Tile-base offsets relative to the dest counter start set by _llk_math_eltwise_binary_sfpu_params_.
    // Per-row stride comes from ADDR_MOD_6's dest.incr=2 on SFPSTORE, not from these offsets.
    const std::uint32_t offset0 = (dst_index_in0 * 32) << 1;
    const std::uint32_t offset1 = (dst_index_in1 * 32) << 1;
    const std::uint32_t offset2 = (dst_index_out * 32) << 1;

    lltt::record(0, BINARY_MAX_MIN_REPLAY_LEN);
    TT_SFPLOAD(p_sfpu::LREG0, SFPMEM_MODE, ADDR_MOD_7, 0 /* done */, offset0);
    TT_SFPLOAD(p_sfpu::LREG1, SFPMEM_MODE, ADDR_MOD_7, 0 /* done */, offset1);

    // Step 1: SM-min/SM-max via SFPSWAP. Correct for any pair where at least one
    // operand is non-negative; inverts ordering for (neg, neg) pairs.
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // 2-cycle
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);  // post-SFPSWAP stall avoidance

    // Step 2: CC-guarded correction swap for (neg, neg) pairs.
    // Successive SFPSETCC calls AND into CC, so after both: CC = (LREG0<0 AND LREG1<0).
    TTI_SFPSETCC(SFPSETCC_INT32_SIGNBIT, p_sfpu::LREG0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(SFPSETCC_INT32_SIGNBIT, p_sfpu::LREG1, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSWAP(
        0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);  // re-swap rows where both negative
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);

    // After step 2: LREG0 = min, LREG1 = max for all sign combinations.
    // ADDR_MOD_6 (dest.incr=2) advances the dest counter by one row pair after each store.
    TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, SFPMEM_MODE, ADDR_MOD_6, 0 /* done */, offset2);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(0, BINARY_MAX_MIN_REPLAY_LEN);
    }
}

}  // namespace sfpu
}  // namespace ckernel
