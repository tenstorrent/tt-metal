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

// Never-true predicate for the terminal `else` of an OP dispatch: keeps the static_assert
// dependent on OP so it only fires for a QuantVariant the dispatch does not handle.
template <QuantVariant>
inline constexpr bool quant_unhandled_variant = false;

// Shared LREG layout across all three quant ops. Each replay processes TWO SFPU passes (row pairs):
//   LREG0 = pass 0 operand A load / running accumulator / result before store
//   LREG1 = pass 0 operand B (fp32 scale)
//   LREG3 = pass 1 operand A load / running accumulator / result before store
//   LREG4 = pass 1 operand B (fp32 scale)
//   LREG2 = zero-point (fp32), loaded once by the matching init
// LREG5-7 stay free. Each init must run before the first call to its kernel.
//
// SIGN_MAGNITUDE_FORMAT gates the SM<->2's-comp casts: the default (false) path
// treats int32 dest content as 2's-complement (UNP_DEST / Int32-L1), while
// STOCH_RND emits sign-magnitude (SMAG32). Inputs/outputs are therefore cast on
// the !SIGN_MAGNITUDE_FORMAT path; the true path stores STOCH_RND output as-is.
//
// The init records once the register-to-register math that turns the loaded
// operands into the result; the kernel replays it for every pair of SFPU passes
// around inline operand loads and result stores.
//
// Each recorded body handles two passes: pass 0 in LREG0/LREG1, pass 1 in LREG3/LREG4,
// with the two chains interleaved instruction by instruction (p0/p1 below). Every
// instruction of a chain (CAST -> CAST -> MAD -> STOCH_RND -> CAST, MAD -> MUL) depends
// on the previous one, so a single-pass body issues each consumer right behind its
// producer; with the other pass's independent instruction in between, the SFPU
// pipeline has one more issue slot to retire the producer before the consumer reads
// it. Per element the instruction sequence and operand order are unchanged.
//
// Per-op slots are non-overlapping and sized to the larger (2's-complement)
// variant (the sign-magnitude variant records fewer instructions into the same
// slot), so one compute kernel can mix all three ops without cross-clobbering:
//   QUANT   (2s-comp, 6)  : SFPMAD p0/p1, STOCH_RND p0/p1, SFPCAST p0/p1
//   QUANT   (sign-m, 4)   : SFPMAD p0/p1, STOCH_RND p0/p1
//   REQUANT (2s-comp, 10) : SFPCAST(in) p0/p1, SFPCAST(int->fp32) p0/p1, SFPMAD p0/p1, STOCH_RND p0/p1,
//                           SFPCAST(out) p0/p1
//   REQUANT (sign-m, 6)   : SFPCAST(int->fp32) p0/p1, SFPMAD p0/p1, STOCH_RND p0/p1
//   DEQUANT (2s-comp, 8)  : SFPCAST(in) p0/p1, SFPCAST(int->fp32) p0/p1, SFPMAD p0/p1, SFPMUL p0/p1
//   DEQUANT (sign-m, 6)   : SFPCAST(int->fp32) p0/p1, SFPMAD p0/p1, SFPMUL p0/p1
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN_2S_COMP = 6;
constexpr std::uint32_t QUANT_REPLAY_LEN_SIGN_MAGN = 4;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t REQUANT_REPLAY_LEN_2S_COMP = 10;
constexpr std::uint32_t REQUANT_REPLAY_LEN_SIGN_MAGN = 6;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_2S_COMP = 8;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_SIGN_MAGN = 6;

static_assert(
    DEQUANT_REPLAY_SLOT + DEQUANT_REPLAY_LEN_2S_COMP <= 32, "quant replay bodies exceed the 32-entry replay bank");

// Pass-pair register plan (see the replay-body comment above).
constexpr std::uint32_t QUANT_PASS0_A = p_sfpu::LREG0;
constexpr std::uint32_t QUANT_PASS0_B = p_sfpu::LREG1;
constexpr std::uint32_t QUANT_PASS1_A = p_sfpu::LREG3;
constexpr std::uint32_t QUANT_PASS1_B = p_sfpu::LREG4;

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

/**
 * @brief Record the register-only compute middle for a quant-family op into its replay slot.
 *
 * Loads the fp32 zero-point into LREG2, then streams the op's compute body into a replay
 * buffer (NoExec: LREG0/LREG1 are undefined at init time). The body is replayed per loop
 * iteration by @ref quant_family. Op math (A = operand, B = fp32 scale, zp = zero-point):
 *   Quant   : out = round_to_int8(A_fp32 * B + zp)
 *   Requant : out = round_to_int8(int32_to_fp32(A) * B + zp)
 *   Dequant : out = (int32_to_fp32(A) + zp) * B   -- caller passes bits of -zero_point
 *
 * @tparam OP: Which quant-family op, values = <Quant/Requant/Dequant>
 * @tparam SIGN_MAGNITUDE_FORMAT: If true, treat int32 dest as SMAG32 and skip the SM<->2's-comp casts.
 * @param zero_point: fp32 bit-pattern of the zero-point (Dequant expects the bits of -zero_point).
 * @note Call once before @ref quant_family with the same OP / SIGN_MAGNITUDE_FORMAT.
 */
template <QuantVariant OP, bool SIGN_MAGNITUDE_FORMAT = false>
inline void quant_family_init(const std::uint32_t zero_point) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, zero_point & 0xFFFF);  // zp low half
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, zero_point >> 16);     // zp high half -> LREG2 = fp32 zp

    constexpr std::uint32_t slot = quant_replay_slot<OP>();
    constexpr std::uint32_t len = quant_replay_len<OP, SIGN_MAGNITUDE_FORMAT>();
    load_replay_buf<slot, len, /*exec_while_loading=*/false>([] {
        // Requant/Dequant take an int32 operand A -> bring it to fp32 first.
        if constexpr (OP == QuantVariant::Requant || OP == QuantVariant::Dequant) {
            if constexpr (!SIGN_MAGNITUDE_FORMAT) {
                // 2's-comp -> sign-mag input, both passes
                TTI_SFPCAST(QUANT_PASS0_A, QUANT_PASS0_A, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
                TTI_SFPCAST(QUANT_PASS1_A, QUANT_PASS1_A, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);
            }
            // int32 (sign-mag) -> fp32 RNE, both passes
            TTI_SFPCAST(QUANT_PASS0_A, QUANT_PASS0_A, 0 /* mod1 */);
            TTI_SFPCAST(QUANT_PASS1_A, QUANT_PASS1_A, 0 /* mod1 */);
        }

        if constexpr (OP == QuantVariant::Quant || OP == QuantVariant::Requant) {
            // A*B + zp, both passes
            TTI_SFPMAD(QUANT_PASS0_A, QUANT_PASS0_B, p_sfpu::LREG2, QUANT_PASS0_A, 0 /* mod1 */);
            TTI_SFPMAD(QUANT_PASS1_A, QUANT_PASS1_B, p_sfpu::LREG2, QUANT_PASS1_A, 0 /* mod1 */);

            // fp32 -> signed int8 (SMAG32 container), round-nearest-even; (1<<3) selects the imm8 descale slot (0 = no
            // descale). Both passes.
            TTI_SFP_STOCH_RND(
                p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
                0 /* imm8 descale */,
                p_sfpu::LCONST_0 /* lreg_b (unused on fp32 path) */,
                QUANT_PASS0_A,
                QUANT_PASS0_A,
                (1 << 3) | p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8);
            TTI_SFP_STOCH_RND(
                p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
                0 /* imm8 descale */,
                p_sfpu::LCONST_0 /* lreg_b (unused on fp32 path) */,
                QUANT_PASS1_A,
                QUANT_PASS1_A,
                (1 << 3) | p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8);

            if constexpr (!SIGN_MAGNITUDE_FORMAT) {
                // sign-mag -> 2's-comp output, both passes
                TTI_SFPCAST(QUANT_PASS0_A, QUANT_PASS0_A, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
                TTI_SFPCAST(QUANT_PASS1_A, QUANT_PASS1_A, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);
            }
        } else {  // Dequant: (A + (-zp)) * B, fp32 output (no round / no output cast)
            // LCONST_1 = 1.0 -> A*1 + (-zp) = A - zp ; then LCONST_0 = 0.0 ignored as +C -> (A - zp) * B
            TTI_SFPMAD(QUANT_PASS0_A, p_sfpu::LCONST_1, p_sfpu::LREG2, QUANT_PASS0_A, 0 /* mod1 */);
            TTI_SFPMAD(QUANT_PASS1_A, p_sfpu::LCONST_1, p_sfpu::LREG2, QUANT_PASS1_A, 0 /* mod1 */);
            TTI_SFPMUL(QUANT_PASS0_A, QUANT_PASS0_B, p_sfpu::LCONST_0, QUANT_PASS0_A, 0 /* mod1 */);
            TTI_SFPMUL(QUANT_PASS1_A, QUANT_PASS1_B, p_sfpu::LCONST_0, QUANT_PASS1_A, 0 /* mod1 */);
        }
    });
}

/**
 * @brief Apply a quant-family op over a tile: per pair of passes load A/B of both, replay the compute middle, store
 * both.
 *
 * Operand A is loaded as fp32 for Quant and int32 for Requant/Dequant; the result is stored as
 * int32 for Quant/Requant and fp32 for Dequant. The runtime-address loads/stores stay inline; the
 * register-only middle recorded by @ref quant_family_init is replayed once per pair of passes.
 *
 * @tparam OP: Which quant-family op, values = <Quant/Requant/Dequant>
 * @tparam ITERATIONS: Number of SFPU passes spanning the tile (default = SFPU_ITERATIONS); must be even.
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

    static_assert(ITERATIONS % 2 == 0, "quant_family processes SFPU passes in pairs");
    constexpr std::uint32_t PASS_ROWS = ckernel::math::SFP_ROWS;  // Dest rows per SFPU pass

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        const std::uint32_t row = d * PASS_ROWS;  // first Dest row of pass d; pass d + 1 is PASS_ROWS further
        TT_SFPLOAD(QUANT_PASS0_A, A_FORMAT, ADDR_MOD_7, 0, in0_offset + row);              // operand A, pass 0
        TT_SFPLOAD(QUANT_PASS0_B, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, in1_offset + row);  // operand B (fp32 scale)
        TT_SFPLOAD(QUANT_PASS1_A, A_FORMAT, ADDR_MOD_7, 0, in0_offset + row + PASS_ROWS);  // operand A, pass 1
        TT_SFPLOAD(QUANT_PASS1_B, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, in1_offset + row + PASS_ROWS);  // operand B

        TTI_REPLAY(slot, len, 0, 0, 0, 0);  // recorded compute middle, both passes

        TT_SFPSTORE(QUANT_PASS0_A, OUT_FORMAT, ADDR_MOD_7, 0, out_offset + row);              // store result, pass 0
        TT_SFPSTORE(QUANT_PASS1_A, OUT_FORMAT, ADDR_MOD_7, 0, out_offset + row + PASS_ROWS);  // store result, pass 1
    }
}

}  // namespace ckernel::sfpu
