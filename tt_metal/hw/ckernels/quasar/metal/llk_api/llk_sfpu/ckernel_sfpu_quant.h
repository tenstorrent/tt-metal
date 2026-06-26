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

inline constexpr bool is_quant_variant(QuantVariant op) {
    return op == QuantVariant::Quant || op == QuantVariant::Requant || op == QuantVariant::Dequant;
}

// Shared LREG layout across all three quant ops:
//   LREG0 = operand A load / running accumulator / result before store
//   LREG1 = operand B (fp32 scale)
//   LREG2 = zero-point (fp32), loaded once by the matching init
// LREG3-7 stay free. Each init must run before the first call to its kernel.
//
// SIGN_MAGNITUDE_FORMAT gates the SM<->2's-comp casts: the default (false) path
// treats int32 dest content as 2's-complement (UNP_DEST / Int32-L1), while
// STOCH_RND emits sign-magnitude (SMAG32). Inputs/outputs are therefore cast on
// the !SIGN_MAGNITUDE_FORMAT path; the true path stores STOCH_RND output as-is.
//
// Replay-buffer optimization (mirrors the Blackhole reference):
// the per-iteration REGISTER-ONLY compute middle of each op (no runtime
// addresses) is recorded ONCE by the matching init via load_replay_buf and
// then replayed per loop iteration with TTI_REPLAY. The runtime-address
// SFPLOADs/SFPSTORE (base + (d<<1)) stay INLINE and are NOT recorded - exactly
// as Blackhole keeps its runtime-address loads/stores inline. Recording uses
// load_mode=1 with execute_while_loading=false (the load_replay_buf default),
// i.e. the body is streamed into the replay buffer WITHOUT executing it at
// record time, so it never reads the undefined LREG0/LREG1 at init.
//
// Per-op slots are non-overlapping and sized to the larger (2's-complement)
// variant; the matching sign-magnitude variant records fewer instructions into
// the same slot and replays the shorter length. Distinct slots between ops let
// a single compute kernel mix all three ops without an init clobbering another
// op's recording.
//   QUANT   (2s-comp, 3) : SFPMAD, STOCH_RND, SFPCAST
//   QUANT   (sign-m, 2)  : SFPMAD, STOCH_RND
//   REQUANT (2s-comp, 5) : SFPCAST(in), SFPCAST(int->fp32), SFPMAD, STOCH_RND, SFPCAST(out)
//   REQUANT (sign-m, 3)  : SFPCAST(int->fp32), SFPMAD, STOCH_RND
//   DEQUANT (2s-comp, 4) : SFPCAST(in), SFPCAST(int->fp32), SFPMAD, SFPMUL
//   DEQUANT (sign-m, 3)  : SFPCAST(int->fp32), SFPMAD, SFPMUL
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN_2S_COMP = 3;
constexpr std::uint32_t QUANT_REPLAY_LEN_SIGN_MAGN = 2;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t REQUANT_REPLAY_LEN_2S_COMP = 5;
constexpr std::uint32_t REQUANT_REPLAY_LEN_SIGN_MAGN = 3;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_2S_COMP = 4;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_SIGN_MAGN = 3;

// Replay slot base (per op) and recorded length (per op, per format) for the
// register-only compute middle.
template <QuantVariant OP>
inline constexpr std::uint32_t quant_replay_slot() {
    if constexpr (OP == QuantVariant::Quant) {
        return QUANT_REPLAY_SLOT;
    } else if constexpr (OP == QuantVariant::Requant) {
        return REQUANT_REPLAY_SLOT;
    } else {
        return DEQUANT_REPLAY_SLOT;
    }
}

template <QuantVariant OP, bool SIGN_MAGNITUDE_FORMAT>
inline constexpr std::uint32_t quant_replay_len() {
    if constexpr (OP == QuantVariant::Quant) {
        return SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;
    } else if constexpr (OP == QuantVariant::Requant) {
        return SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;
    } else {
        return SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;
    }
}

/**
 * @brief Record the register-only compute middle for a quant-family op into its replay slot.
 *
 * Loads the fp32 zero-point into LREG2, then streams the op's compute body into a replay
 * buffer (NoExec: LREG0/LREG1 are undefined at init time). The body is replayed per loop
 * iteration by @ref _quant_family_. Op math (A = operand, B = fp32 scale, zp = zero-point):
 *   Quant   : out = round_to_int8(A_fp32 * B + zp)
 *   Requant : out = round_to_int8(int32_to_fp32(A) * B + zp)
 *   Dequant : out = (int32_to_fp32(A) + zp) * B   -- caller passes bits of -zero_point
 *
 * @tparam OP: Which quant-family op, values = <Quant/Requant/Dequant>
 * @tparam SIGN_MAGNITUDE_FORMAT: If true, treat int32 dest as SMAG32 and skip the SM<->2's-comp casts.
 * @param zero_point: fp32 bit-pattern of the zero-point (Dequant expects the bits of -zero_point).
 * @note Call once before @ref _quant_family_ with the same OP / SIGN_MAGNITUDE_FORMAT.
 */
template <QuantVariant OP, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _quant_family_init_(const std::uint32_t zero_point) {
    static_assert(is_quant_variant(OP), "_quant_family_init_: OP must be Quant/Requant/Dequant");

    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, zero_point & 0xFFFF);  // zp low half
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, zero_point >> 16);     // zp high half -> LREG2 = fp32 zp

    constexpr std::uint32_t slot = quant_replay_slot<OP>();
    constexpr std::uint32_t len = quant_replay_len<OP, SIGN_MAGNITUDE_FORMAT>();
    load_replay_buf<slot, len, /*exec_while_loading=*/false>([] {
        // Requant/Dequant take an int32 operand A -> bring it to fp32 first.
        if constexpr (OP == QuantVariant::Requant || OP == QuantVariant::Dequant) {
            if constexpr (!SIGN_MAGNITUDE_FORMAT) {
                TTI_SFPCAST(
                    p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag input
            }
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32(sign-mag) -> fp32 RNE */);
        }

        if constexpr (OP == QuantVariant::Quant || OP == QuantVariant::Requant) {
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);  // A*B + zp

            // fp32 -> signed int8 (SMAG32 container), round-nearest-even; (1<<3) selects the imm8 descale slot (0 = no
            // descale)
            TTI_SFP_STOCH_RND(
                p_sfpu::sfp_stochrnd_rnd_mod::NearEven,
                0 /* imm8 descale */,
                0 /* lreg_b (unused on fp32 path) */,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                (1 << 3) | p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8);

            if constexpr (!SIGN_MAGNITUDE_FORMAT) {
                TTI_SFPCAST(
                    p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);  // sign-mag -> 2's-comp output
            }
        } else {  // Dequant: (A + (-zp)) * B, fp32 output (no round / no output cast)
            // LCONST_1 = 1.0 -> A*1 + (-zp) = A - zp ; then LCONST_0 = 0.0 ignored as +C -> (A - zp) * B
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);
        }
    });
}

/**
 * @brief Apply a quant-family op over a tile: per iteration load A/B, replay the compute middle, store.
 *
 * Operand A is loaded as fp32 for Quant and int32 for Requant/Dequant; the result is stored as
 * int32 for Quant/Requant and fp32 for Dequant. The runtime-address loads/store stay inline; the
 * register-only middle recorded by @ref _quant_family_init_ is replayed each iteration.
 *
 * @tparam OP: Which quant-family op, values = <Quant/Requant/Dequant>
 * @tparam ITERATIONS: Number of SFPU passes spanning the tile (default = SFPU_ITERATIONS).
 * @tparam SIGN_MAGNITUDE_FORMAT: Must match the value passed to @ref _quant_family_init_.
 * @param dst_index_in0,dst_index_in1,dst_index_out: Dest row offsets of operand A, operand B, and the result.
 * @note Call @ref _quant_family_init_ with the same OP / SIGN_MAGNITUDE_FORMAT before this.
 */
template <QuantVariant OP, int ITERATIONS = SFPU_ITERATIONS, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _quant_family_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_quant_variant(OP), "_quant_family_: OP must be Quant/Requant/Dequant");

    constexpr std::uint32_t slot = quant_replay_slot<OP>();
    constexpr std::uint32_t len = quant_replay_len<OP, SIGN_MAGNITUDE_FORMAT>();
    constexpr auto A_FORMAT = (OP == QuantVariant::Quant) ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::INT32;
    constexpr auto OUT_FORMAT = (OP == QuantVariant::Dequant) ? p_sfpu::sfpmem::FP32 : p_sfpu::sfpmem::INT32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, A_FORMAT, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        TTI_REPLAY(slot, len, 0, 0, 0, 0);  // recorded compute middle

        TT_SFPSTORE(p_sfpu::LREG0, OUT_FORMAT, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store result
    }
}

}  // namespace ckernel::sfpu
