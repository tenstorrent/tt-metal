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
// addresses) is recorded ONCE by the matching _init_ via load_replay_buf and
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

// ---- quant: out_int32 = round_to_int8(A_fp32 * B_fp32 + zero_point) ----

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _quant_int32_init_(const std::uint32_t zero_point /* fp32 bits */) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, zero_point & 0xFFFF);  // zp low half
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, zero_point >> 16);     // zp high half -> LREG2 = fp32 zp

    // Record the register-only compute middle once (NoExec: LREG0/LREG1 are
    // undefined at init time). Replayed per iteration by _quant_int32_.
    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;
    load_replay_buf<QUANT_REPLAY_SLOT, REPLAY_LEN, /*exec_while_loading=*/false>([] {
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);  // LREG0 = A*B + zp

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
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                p_sfpu::sfp_sfpcast_mod::SM32_TO_2SC);  // sign-mag -> 2's-comp for INT32 store
        }
    });
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _quant_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A (fp32)
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        TTI_REPLAY(QUANT_REPLAY_SLOT, REPLAY_LEN, 0, 0, 0, 0);  // MAD + STOCH_RND + (CAST)

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store int32
    }
}

// ---- requant: out_int32 = round_to_int8(int32_to_fp32(A) * B_fp32 + zero_point) ----

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _requant_int32_init_(const std::uint32_t zero_point /* fp32 bits */) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, zero_point & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, zero_point >> 16);  // LREG2 = fp32 zp

    // Record the register-only compute middle once (NoExec). Replayed per
    // iteration by _requant_int32_.
    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;
    load_replay_buf<REQUANT_REPLAY_SLOT, REPLAY_LEN, /*exec_while_loading=*/false>([] {
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(
                p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag input
        }
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32(sign-mag) -> fp32 RNE */);

        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);  // LREG0 = A_fp*B + zp

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
    });
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _requant_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A (int32)
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        TTI_REPLAY(REQUANT_REPLAY_SLOT, REPLAY_LEN, 0, 0, 0, 0);

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store int32
    }
}

// ---- dequant: out_fp32 = (int32_to_fp32(A) - zero_point) * B_fp32 ----
// init negates zp so the body computes (A + (-zp)) * B.

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _dequant_int32_init_(const std::uint32_t neg_zero_point /* fp32 bits of -zp */) {
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, neg_zero_point & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, neg_zero_point >> 16);  // LREG2 = fp32 (-zp)

    // Record the register-only compute middle once (NoExec). Replayed per
    // iteration by _dequant_int32_.
    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;
    load_replay_buf<DEQUANT_REPLAY_SLOT, REPLAY_LEN, /*exec_while_loading=*/false>([] {
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            TTI_SFPCAST(
                p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::sfp_sfpcast_mod::TWO_SC_TO_SM);  // 2's-comp -> sign-mag input
        }
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0 /* mod1: int32(sign-mag) -> fp32 RNE */);

        // LCONST_1 = 1.0 -> A*1 + (-zp) = A - zp ; then LCONST_0 = 0.0 ignored as +C -> (A - zp) * B
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /* mod1 */);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /* mod1 */);
    });
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _dequant_int32_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, dst_index_in0 + (d << 1));  // operand A (int32)
        TT_SFPLOAD(
            p_sfpu::LREG1, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_in1 + (d << 1));  // operand B (fp32 scale)

        TTI_REPLAY(DEQUANT_REPLAY_SLOT, REPLAY_LEN, 0, 0, 0, 0);

        TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0, dst_index_out + (d << 1));  // store fp32
    }
}

}  // namespace ckernel::sfpu
