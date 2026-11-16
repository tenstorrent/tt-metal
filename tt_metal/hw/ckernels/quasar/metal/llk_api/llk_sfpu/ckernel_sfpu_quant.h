// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// The three quant-family binary ops, selected at compile time by one op-templated
// init/execute pair below (quant / requant / dequant).
enum class QuantVariant : std::uint8_t { Quant, Requant, Dequant };

// Round to nearest even, then saturate to the int8 range [-128, 127].
//
// SFP_STOCH_RND's fp32->int8 mode saturates at +/-127 and cannot represent -128, and a negative
// whose magnitude rounds to zero leaves it as sign-magnitude -0, which the following cast reads
// back as -1. Quant and requant therefore round the way Blackhole does. Adding the magic constant
// 1.5 * 2^23 puts the value where the fp32 spacing is 1, so the add itself is round-to-nearest-even
// for |v| < 2^22:
//     m = RNE(RNE_MAGIC + zp_hi + v) ,  t = RNE_MAGIC - zp_lo ,  bits(m) - bits(t) = RNE(v) + zp
// zp_hi is a multiple of 4 near zp, so zp_lo = zp - zp_hi stays in [-2, 2] and adding zp_hi before
// the rounding add does not change how a tie rounds. Exact for |zp| <= 2^24. Quant puts its zero
// point through this split, so it lands after the round: q = RNE(A * scale) + zp. Requant adds its
// input zero point before the round and puts its output zero point through the split. Both match
// the Blackhole contract, which the shared WH/BH golden pins at exact halves.
//
// The biased m is then clamped to [t - 128, t + 127] before the subtract, which is the full int8
// range. Quasar's SFPSWAP writes both operands and mode 9 (VEC_MAX_MIN) is reserved, so both clamps
// use ALL_ROWS_MAX (mode 1, smaller result to lreg_dest). The upper bound sits in programmable
// constant RG13, whose write-back is dropped. The lower bound cannot: a destination register >= 12
// is macro capture, so it lives in LREG4 and is copied into scratch LREG3 on every iteration.
// RG11 is left alone; writing it would clobber LCONST_neg1.
constexpr std::uint32_t RNE_MAGIC_FP32 = 0x4b400000u;  // 12582912.0f = 1.5 * 2^23
constexpr std::uint32_t ZP_SPLIT_FP32 = 0x4c400000u;   // 50331648.0f = 1.5 * 2^25
constexpr int QUANT_CLAMP_LO = -128;
constexpr int QUANT_CLAMP_HI = 127;
constexpr std::uint32_t QUANT_SFPSWAP_IMM12_FP32 = 0x1;  // imm12[0] = 1 selects the fp32 compare
// Register-file index of the programmable constant holding the upper bound.
constexpr std::uint32_t QUANT_CLAMP_HI_LREG = 13;
constexpr std::uint32_t QUANT_IADD_MOD_IMM = sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE;
constexpr std::uint32_t QUANT_IADD_MOD_ADD =
    sfpi::SFPIADD_MOD1_ARG_LREG_DST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;
constexpr std::uint32_t QUANT_IADD_MOD_NEG =
    sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | p_sfpu::sfp_binary_mod::SFPIADD_DISABLE_CC;

// Never-true predicate for the terminal `else` of an OP dispatch: keeps the static_assert
// dependent on OP so it only fires for a QuantVariant the dispatch does not handle.
template <QuantVariant>
inline constexpr bool quant_unhandled_variant = false;

// Shared LREG layout. Each init must run before the first call to its kernel; the
// constants below are shared, so mixing ops means re-running the matching init.
//   LREG0 = operand A / running value / result
//   LREG1 = operand B (fp32 scale). Also scratch while an init builds the RNE constants.
//   LREG2 = the fp32 zero point added before the round (requant's input zp, or dequant's
//           negated zp). Quant's zero point rides the bias instead, so quant leaves this free.
//   LREG3 = per-iteration scratch for the lower clamp (holds the min after SFPSWAP)
//   LREG4 = lower clamp bound, copied into LREG3 every iteration
//   LREG5 = RNE_MAGIC + zp_hi of whichever zero point the bias carries
//   LREG6 = -bits(t), built from quant's zero point or requant's output zero point
//   RG13  = upper clamp bound (programmable constant; SFPSWAP write-back is dropped)
//
// SIGN_MAGNITUDE_FORMAT selects the int32 representation in Dest. The default (false)
// is two's complement, which is what unpack-to-dest delivers. The true path skips the
// input cast and converts the two's-complement result back to sign-magnitude before
// the store. Dequant never rounds, so it only casts its int32 input.
//
// The init records the register-only middle once; the kernel replays it around the
// runtime-address loads and the store. Slots are non-overlapping and sized to the
// longer of the two formats. Replay-bank 0's switch slot is 31, and these three
// bodies stay below it. Counts, in emission order:
//   QUANT   (2s-comp, 5) : SFPMAD(A*B + bias), <clamp>
//   QUANT   (sign-m, 6) : SFPMAD(A*B + bias), <clamp>, SFPCAST(out)
//   REQUANT (2s-comp, 8): SFPCAST(in), SFPCAST(int->fp32), SFPMAD, SFPADD, <clamp>
//   REQUANT (sign-m, 8): SFPCAST(int->fp32), SFPMAD, SFPADD, <clamp>, SFPCAST(out)
//   DEQUANT (2s-comp, 4) : SFPCAST(in), SFPCAST(int->fp32), SFPMAD, SFPMUL
//   DEQUANT (sign-m, 3)  : SFPCAST(int->fp32), SFPMAD, SFPMUL
// <clamp> is 4 instructions: SFPMOV, SFPSWAP(hi), SFPSWAP(lo), SFPIADD.
// The SFPMOV is independent of the arithmetic before it, so it fills the TEN-4581 shadow a
// 2-cycle op would otherwise need before SFPSWAP. No NOP follows either swap:
// SFPSWAP stalls the next SFPU op itself, same as gcd and max_pool_indices.
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN_2S_COMP = 5;
constexpr std::uint32_t QUANT_REPLAY_LEN_SIGN_MAGN = 6;
constexpr std::uint32_t QUANT_REPLAY_LEN_MAX = QUANT_REPLAY_LEN_SIGN_MAGN;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t REQUANT_REPLAY_LEN_2S_COMP = 8;
constexpr std::uint32_t REQUANT_REPLAY_LEN_SIGN_MAGN = 8;
constexpr std::uint32_t REQUANT_REPLAY_LEN_MAX = REQUANT_REPLAY_LEN_2S_COMP;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_2S_COMP = 4;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_SIGN_MAGN = 3;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_MAX = DEQUANT_REPLAY_LEN_2S_COMP;

// Bank 0's last slot flips the replay bank; keep the recorded bodies out of it.
constexpr std::uint32_t QUANT_REPLAY_BANK_SWITCH_SLOT = 31;
static_assert(
    DEQUANT_REPLAY_SLOT + DEQUANT_REPLAY_LEN_MAX <= QUANT_REPLAY_BANK_SWITCH_SLOT,
    "quant-family replay bodies overlap the bank-switch slot");

// Replay slot base (per op) and recorded length (per op, per format) for the
// register-only compute middle.
template <QuantVariant OP>
inline constexpr std::uint32_t quant_replay_slot() {
    if constexpr (OP == QuantVariant::Quant) {
        return QUANT_REPLAY_SLOT;
    } else if constexpr (OP == QuantVariant::Requant) {
        return REQUANT_REPLAY_SLOT;
    } else if constexpr (OP == QuantVariant::Dequant) {
        return DEQUANT_REPLAY_SLOT;
    } else {
        static_assert(quant_unhandled_variant<OP>, "quant_replay_slot: unhandled QuantVariant");
    }
}

template <QuantVariant OP, bool SIGN_MAGNITUDE_FORMAT>
inline constexpr std::uint32_t quant_replay_len() {
    if constexpr (OP == QuantVariant::Quant) {
        return SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;
    } else if constexpr (OP == QuantVariant::Requant) {
        return SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;
    } else if constexpr (OP == QuantVariant::Dequant) {
        return SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;
    } else {
        static_assert(quant_unhandled_variant<OP>, "quant_replay_len: unhandled QuantVariant");
    }
}

__attribute__((always_inline)) inline void _quant_load_imm32_(const std::uint32_t lreg, const std::uint32_t bits) {
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
}

// T_LREG holds the fp32 zero point on entry and -bits(t) on exit. LREG5 becomes
// RNE_MAGIC + zp_hi, LREG4 the lower clamp bound, and RG13 the upper one.
// LREG0 and LREG1 are scratch.
template <std::uint32_t T_LREG>
__attribute__((always_inline)) inline void _rne_clamp_init_() {
    constexpr std::uint32_t lo_imm = static_cast<std::uint32_t>(QUANT_CLAMP_LO) & 0xFFFu;
    constexpr std::uint32_t hi_imm = static_cast<std::uint32_t>(QUANT_CLAMP_HI) & 0xFFFu;

    _quant_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
    _quant_load_imm32_(p_sfpu::LREG1, ZP_SPLIT_FP32);
    // LREG0 = zp_hi = fl(zp + ZP_SPLIT) - ZP_SPLIT. The ulp at ZP_SPLIT is 4, so zp_hi is a multiple of 4.
    TTI_SFPMAD(T_LREG, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0 /* mod1 */);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1 */);
    // T = zp_lo = zp - zp_hi, then t = RNE_MAGIC - zp_lo.
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_neg1, T_LREG, T_LREG, 0 /* mod1 */);
    TTI_SFPMAD(T_LREG, p_sfpu::LCONST_neg1, p_sfpu::LREG5, T_LREG, 0 /* mod1 */);
    // LREG5 = RNE_MAGIC + zp_hi. The rounding add uses this on its own, not folded into A*B.
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG5, 0 /* mod1 */);

    // bits(t) + LO. A GPR, because an SFPSWAP destination >= 12 is macro capture rather than a write.
    TTI_SFPIADD(lo_imm, T_LREG, p_sfpu::LREG0, QUANT_IADD_MOD_IMM);
    TTI_SFPMOV(p_sfpu::LREG0, p_sfpu::LREG4, 0 /* plain copy */);
    // bits(t) + HI, parked in RG13. SFPSWAP writes it back, and that write is dropped.
    TTI_SFPIADD(hi_imm, T_LREG, p_sfpu::LREG0, QUANT_IADD_MOD_IMM);
    TTI_SFPCONFIG(0, p_sfpconfig::PROG_CONST_LREG13, 0);
    TTI_SFPNOP(0, 0, 0);  // nothing may issue the cycle after SFPCONFIG

    TTI_SFPIADD(0, p_sfpu::LCONST_0, T_LREG, QUANT_IADD_MOD_NEG);  // T = -bits(t)
}

// Four-instruction clamp of the biased fp32 value in LREG0 down to a two's-complement int8.
// BIAS_NEG_LREG holds -bits(t). LREG4 holds the lower bound; LREG3 absorbs the swap write-back.
// The copy does not read the add's result, so it occupies the TEN-4581 shadow before the first
// swap. No NOP follows a swap: SFPSWAP stalls the next SFPU op itself (TEN-4581).
template <std::uint32_t BIAS_NEG_LREG>
__attribute__((always_inline)) inline void _quant_clamp_body_() {
    TTI_SFPMOV(p_sfpu::LREG4, p_sfpu::LREG3, 0 /* plain copy */);  // refresh lo; fills the add→swap shadow
    // LREG0 = min(LREG0, hi). RG13 takes the max and drops the write.
    TTI_SFPSWAP(QUANT_SFPSWAP_IMM12_FP32, QUANT_CLAMP_HI_LREG, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
    // LREG0 = max(LREG0, lo). LREG0 has to be lreg_c so it receives the larger operand.
    TTI_SFPSWAP(QUANT_SFPSWAP_IMM12_FP32, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPIADD(0, BIAS_NEG_LREG, p_sfpu::LREG0, QUANT_IADD_MOD_ADD);  // bits(m) - bits(t)
}

/**
 * @brief Record the register-only compute middle for a quant-family op into its replay slot.
 *
 * Streams the op's compute body into a replay buffer (NoExec: LREG0/LREG1 are undefined at init
 * time). The body is replayed per loop iteration by @ref quant_family. Op math (A = operand,
 * B = fp32 scale, zp = zero-point), with the integer result saturated to [-128, 127]:
 *   Quant   : out = round_to_nearest_even(A_fp32 * B) + zp
 *   Requant : out = round_to_nearest_even(int32_to_fp32(A) * B + zp) + out_zp
 *   Dequant : out = (int32_to_fp32(A) + zp) * B   -- caller passes bits of -zero_point
 *
 * @tparam OP: Which quant-family op, values = <Quant/Requant/Dequant>
 * @tparam SIGN_MAGNITUDE_FORMAT: If true, int32 Dest values are sign-magnitude. Quant and requant
 *         then cast the two's-complement result back to sign-magnitude; requant and dequant skip
 *         the input cast.
 * @param zero_point: fp32 bit-pattern of the zero-point (Dequant expects the bits of -zero_point).
 * @param out_zero_point: fp32 bit-pattern of Requant's output zero-point; ignored by Quant/Dequant.
 * @note Call once before @ref quant_family with the same OP / SIGN_MAGNITUDE_FORMAT.
 */
template <QuantVariant OP, bool SIGN_MAGNITUDE_FORMAT = false>
inline void quant_family_init(const std::uint32_t zero_point, const std::uint32_t out_zero_point = 0) {
    if constexpr (OP == QuantVariant::Quant) {
        // The zero point is split into the bias, which puts it after the round: q = RNE(A*B) + zp.
        // Nothing adds it before the round, so LREG2 goes unused.
        _quant_load_imm32_(p_sfpu::LREG6, zero_point);
        _rne_clamp_init_<p_sfpu::LREG6>();
    } else if constexpr (OP == QuantVariant::Requant) {
        // LREG2 is the input zero point, added before the round. The output zero point is what
        // gets split into the bias.
        _quant_load_imm32_(p_sfpu::LREG2, zero_point);
        _quant_load_imm32_(p_sfpu::LREG6, out_zero_point);
        _rne_clamp_init_<p_sfpu::LREG6>();
    } else {
        _quant_load_imm32_(p_sfpu::LREG2, zero_point);
    }

    constexpr std::uint32_t slot = quant_replay_slot<OP>();
    constexpr std::uint32_t len = quant_replay_len<OP, SIGN_MAGNITUDE_FORMAT>();
    load_replay_buf<slot, len, /*exec_while_loading=*/false>([] {
        if constexpr (OP == QuantVariant::Dequant) {
            if constexpr (!SIGN_MAGNITUDE_FORMAT) {
                TTI_SFPCAST(
                    p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag
            }
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32 sign-mag -> fp32 RNE */);
            // LCONST_1 = 1.0 -> A + (-zp); LCONST_0 = 0.0 is ignored as +C -> (A - zp) * B.
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);
        } else if constexpr (OP == QuantVariant::Quant) {
            // One MAD, as on Blackhole: the product is added to the bias unrounded, so the single
            // rounding the result sees is the magic-constant one. q = RNE(A*B) + zp.
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /* mod1 */);
            _quant_clamp_body_<p_sfpu::LREG6>();
            if constexpr (SIGN_MAGNITUDE_FORMAT) {
                TTI_SFPCAST(
                    p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag
            }
        } else if constexpr (OP == QuantVariant::Requant) {
            if constexpr (!SIGN_MAGNITUDE_FORMAT) {
                TTI_SFPCAST(
                    p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag
            }
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32 sign-mag -> fp32 RNE */);
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);  // A*B + zp
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /* mod1 */);
            _quant_clamp_body_<p_sfpu::LREG6>();
            if constexpr (SIGN_MAGNITUDE_FORMAT) {
                TTI_SFPCAST(
                    p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag
            }
        } else {
            static_assert(quant_unhandled_variant<OP>, "quant_family_init: unhandled QuantVariant");
        }
    });
}

/**
 * @brief Apply a quant-family op over a tile: per iteration load A/B, replay the compute middle, store.
 *
 * Operand A is loaded as fp32 for Quant and int32 for Requant/Dequant; the result is stored as
 * int32 for Quant/Requant and fp32 for Dequant. The runtime-address loads/store stay inline; the
 * register-only middle recorded by @ref quant_family_init is replayed each iteration.
 *
 * @tparam OP: Which quant-family op, values = <Quant/Requant/Dequant>
 * @tparam ITERATIONS: Number of SFPU passes spanning the tile (default = SFPU_ITERATIONS).
 * @tparam SIGN_MAGNITUDE_FORMAT: Must match the value passed to @ref quant_family_init.
 * @tparam TILE_SHAPE: Dest tile shape used to scale the tile indices into row offsets.
 * @param dst_index_in0,dst_index_in1,dst_index_out: Dest tile indices of operand A, operand B, and the result.
 * @note Call @ref quant_family_init with the same OP / SIGN_MAGNITUDE_FORMAT before this.
 */
template <
    QuantVariant OP,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void quant_family(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t slot = quant_replay_slot<OP>();
    constexpr std::uint32_t len = quant_replay_len<OP, SIGN_MAGNITUDE_FORMAT>();
    constexpr auto A_FORMAT = (OP == QuantVariant::Quant) ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::INT32;
    constexpr auto OUT_FORMAT = (OP == QuantVariant::Dequant) ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::INT32;

    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in0_offset = dst_index_in0 * tile_stride;
    const std::uint32_t in1_offset = dst_index_in1 * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, A_FORMAT, ADDR_MOD_7, 0, in0_offset + (d << 1));  // operand A
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, in1_offset + (d << 1));  // operand B (fp32 scale)

        TTI_REPLAY(slot, len, 0, 0, 0, 0);  // recorded compute middle

        TT_SFPSTORE(p_sfpu::LREG0, OUT_FORMAT, ADDR_MOD_7, 0, out_offset + (d << 1));  // store result
    }
}

// ------------------------------------------------------------------------------------------------
// Compute API surface
//
// api/compute/quantization.h reaches for the same ckernel::sfpu names on every architecture, so the
// wrappers below give the kernels above their WH/BH spelling instead of making that header branch on
// the arch. Only the Int32-output endpoints appear here: Quasar has no 8-bit pack or unbias kernel,
// and quantization.h leaves those wrappers undeclared on this arch.
//
// The template parameters exist to match the WH/BH signatures. SIGN_MAGNITUDE_FORMAT is forwarded;
// the Compute API leaves it false, which is what unpack-to-dest delivers (raw 32-bit words, i.e.
// two's-complement Int32). The rest are pinned to the one combination these kernels implement.
// ------------------------------------------------------------------------------------------------

template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32>
inline void quant_init(const std::uint32_t zero_point) {
    static_assert(OUTPUT_FORMAT == DataFormat::Int32, "Quasar quant_init supports an Int32 output only");
    quant_family_init<QuantVariant::Quant, SIGN_MAGNITUDE_FORMAT>(zero_point);
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32,
    bool INT8_INPUT = false>
inline void requant_init(const std::uint32_t zero_point, const std::uint32_t out_zero_point = 0) {
    static_assert(OUTPUT_FORMAT == DataFormat::Int32, "Quasar requant_init supports an Int32 output only");
    static_assert(!INT8_INPUT, "Quasar requant_init has no Int8 input path");
    quant_family_init<QuantVariant::Requant, SIGN_MAGNITUDE_FORMAT>(zero_point, out_zero_point);
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void dequant_init(const std::uint32_t zero_point) {
    static_assert(!INT8_INPUT, "Quasar dequant_init has no Int8 input path");
    quant_family_init<QuantVariant::Dequant, SIGN_MAGNITUDE_FORMAT>(zero_point);
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_quant_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    quant_family<QuantVariant::Quant, ITERATIONS, SIGN_MAGNITUDE_FORMAT, TILE_SHAPE>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    bool INT8_INPUT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_requant_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(!INT8_INPUT, "Quasar calculate_requant_int32 has no Int8 input path");
    quant_family<QuantVariant::Requant, ITERATIONS, SIGN_MAGNITUDE_FORMAT, TILE_SHAPE>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false,
    bool INT8_INPUT = false,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void calculate_dequant_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(!INT8_INPUT, "Quasar calculate_dequant_int32 has no Int8 input path");
    quant_family<QuantVariant::Dequant, ITERATIONS, SIGN_MAGNITUDE_FORMAT, TILE_SHAPE>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

}  // namespace ckernel::sfpu
