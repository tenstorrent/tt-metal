// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Replay-buffer slots for the per-iteration register-only compute bodies of
// the quant / requant / dequant kernels. The body is recorded once by each
// op's _init_{quant,requant,dequant}_int32_ and then replayed by every
// invocation of the matching _{quant,requant,dequant}_int32_ kernel.
// Unlike the Blackhole port, the recorded body does not depend on
// SIGN_MAGNITUDE_FORMAT: WH performs the int32 sign-magnitude <-> 2's
// complement conversion in the SFPLOAD/SFPSTORE instr_mod0 field
// (INT32 = 4 vs INT32_2S_COMP = 12), and those load/store instructions are
// outside the replay window. So one slot per kernel suffices.
//
// Distinct slots between kernels are required so a single compute kernel
// can mix all three ops without each init clobbering the others' recordings.
//
// Body content (see the inits for the exact emission order):
//   QUANT   (3) : SFPMAD, SFPNOP, STOCH_RND
//   REQUANT (4) : SFPCAST(int->fp32), SFPMAD, SFPNOP, STOCH_RND
//   DEQUANT (5) : SFPCAST(int->fp32), SFPADD, SFPNOP, SFPMUL, SFPNOP
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN = 3;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN;
constexpr std::uint32_t REQUANT_REPLAY_LEN = 4;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN;
constexpr std::uint32_t DEQUANT_REPLAY_LEN = 5;

// Configure the SFPU "dest += 2" addr_mod slot. With math::set_addr_mod_base()
// active, the SFPU addr_mod field "2" indexes real config slot ADDR_MOD_6,
// so writing this slot is what the per-iteration SFPSTOREs below pick up
// via ADDR_MOD_2. The addr-mod base is flipped on by every binary-SFPU
// dispatch: _llk_math_eltwise_binary_sfpu_params_ (in
// llk_math_eltwise_binary_sfpu_params.h) calls _llk_math_eltwise_sfpu_start_
// (in llk_math_eltwise_sfpu_common.h), which in turn calls
// math::set_addr_mod_base() before invoking the kernel body.
//
// The matching "no increment" slot used for the SFPLOADs (real ADDR_MOD_7,
// addressed as ADDR_MOD_3 from the SFPU instructions) is already programmed
// to {0,0,0} by eltwise_binary_sfpu_configure_addrmod() in the LLK init, so
// it doesn't need to be set here.
//
// quant_int32 isn't in the LLK init's "configure ADDR_MOD_6 with dest+=2"
// allow-list (that list covers mul_int32 / max / min / cmp ops), so we have
// to program it ourselves. The dest auto-increment of one SFPU dst row
// (sfpi::SFP_DESTREG_STRIDE == 2 dst-address units) is what walks dst_reg
// through the face's 4-row x 8-col blocks. Called once by each
// _init_{quant,requant,dequant}_int32_ since the addrmod state is per-tensix
// and only needs to be set up once. Replaces sfpi::dst_reg++ in the kernel
// bodies and lets each loop be purely TTI-issued.
inline void _quant_kernels_configure_dest_incr_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = sfpi::SFP_DESTREG_STRIDE},
    }
        .set(ADDR_MOD_6);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_quant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input (fp32).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_quant_int32_.
    // Output is int32 scaled to int8 range (sign-magnitude or 2's-complement
    // depending on SIGN_MAGNITUDE_FORMAT - the conversion is done by SFPSTORE).
    //
    // Tile layout in Dest: each tile occupies 64 dest-address units. Each
    // SFPLOAD/SFPSTORE moves 4 dest rows x 8 SFPU lanes, so advancing dst_reg
    // by +2 between iterations walks one face's eight 4-row x 8-col blocks
    // (= one full call site, ITERATIONS == 8).
    //
    // The replay-buffer body at QUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_quant_int32_, which must run before the
    // first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore out_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: inline TT_SFPLOADs (variable addresses can't live inside
    // the replay buffer because TT_* macros write to instrn_buffer[0]), replay
    // the recorded compute, then SFPSTORE under ADDR_MOD_2 (real slot 6) which
    // also auto-advances dst_reg by 2 for the next iteration's loads.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_3, in0_off);  // operand A (fp32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN);                        // MAD + SFPNOP + STOCH_RND
        TT_SFPSTORE(p_sfpu::LREG0, out_mode, ADDR_MOD_2, out_off);                // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_requant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input to requant (int32, sign-magnitude or 2's complement bits).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_requant_int32_.
    // Output is int32 scaled to int8 range.
    //
    // The int32 in/out format conversion is done by the SFPLOAD/SFPSTORE
    // instr_mod0 field (INT32 vs INT32_2S_COMP); the recorded compute body
    // is identical for both SIGN_MAGNITUDE_FORMAT variants.
    //
    // The replay-buffer body at REQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_requant_int32_, which must run before the
    // first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore int_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute
    // (the int->fp cast doesn't touch LREG1 so reordering is safe), replay
    // the recorded body, then SFPSTORE under ADDR_MOD_2 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, int_mode, ADDR_MOD_3, in0_off);  // operand A (int32 -> sign-magn LREG0)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN);                    // CAST + MAD + SFPNOP + STOCH_RND
        TT_SFPSTORE(p_sfpu::LREG0, int_mode, ADDR_MOD_2, out_off);  // store (sign-magn -> int_mode bits) + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_dequant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A[LREG0] is input to dequant (int32, sign-magnitude or 2's complement bits).
    // Operand B[LREG1] is scaling factor (fp32).
    // LREG2 holds the (negated) zero-point constant loaded by _init_dequant_int32_;
    // i.e. the formula computed is (A + LREG2) * B, which is (A - zero_point) * B
    // when the caller passes -zero_point through the init.
    //
    // SFPLOAD's instr_mod0 normalizes both int32 representations to LREG0
    // sign-magnitude before the recorded compute runs, so the body is the
    // same for both SIGN_MAGNITUDE_FORMAT variants.
    //
    // The replay-buffer body at DEQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_dequant_int32_, which must run before the
    // first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore in_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute,
    // replay the body, then SFPSTORE under ADDR_MOD_2 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, in_mode, ADDR_MOD_3, in0_off);  // operand A (int32 -> sign-magn LREG0)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);   // operand B (fp32 scaler)
        lltt::replay(DEQUANT_REPLAY_SLOT, DEQUANT_REPLAY_LEN);                     // CAST + ADD + SFPNOP + MUL + SFPNOP
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_2, out_off);  // store fp32 + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false>
void quant_init(const uint zero_point) {
    // One-time setup for calculate_quant:
    //   1. load the fp32 zero-point constant into LREG2;
    //   2. program ADDR_MOD_6 with dest+=2 for the per-iteration SFPSTORE;
    //   3. record the register-only compute body into the SFPU replay buffer
    //      under QUANT_REPLAY_SLOT (NoExec - we don't want to issue SFPMAD/
    //      STOCH_RND against undefined LREG0/LREG1 contents at record time).
    // Subsequent _quant_int32_ calls replay the recorded body, shrinking the
    // unrolled binary from ~ITERATIONS*REPLAY_LEN body instructions down to
    // one replay invocation per iteration.
    //
    // The recorded body is the same for both SIGN_MAGNITUDE_FORMAT variants
    // (the int32 in/out format conversion is done by SFPLOAD/SFPSTORE
    // instr_mod0 outside the replay window).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN);
    {
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPMAD has a 2-cycle write latency on LREG0 and SFP_STOCH_RND below
        // reads LREG0, so exactly one SFPU pipeline bubble is required. Use
        // TTI_SFPNOP (SFPU NOP) rather than the generic Tensix TTI_NOP so the
        // bubble lands in the SFPU pipe and so the recorded body contains
        // only SFPU-pipe opcodes (a hard requirement for replay-buffer
        // playback - the replay buffer feeds the SFPU pipe directly).
        TTI_SFPNOP;
        // fp32 -> int sign-magnitude. LCONST_0 (LREG9) is the HW-provided 0.0
        // used as the zero descale.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0 /*imm8*/,
            p_sfpu::LCONST_0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false>
void requant_init(const uint zero_point) {
    // One-time setup for calculate_requant; see quant_init for the
    // record/replay rationale. Loads the zero point into LREG2, programs
    // ADDR_MOD_6 with dest+=2, then records the register-only compute into
    // REQUANT_REPLAY_SLOT (NoExec).
    //
    // The recorded body is identical for both SIGN_MAGNITUDE_FORMAT variants;
    // the int32 in/out conversion happens in SFPLOAD/SFPSTORE instr_mod0.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN);
    {
        // int32 sign-magnitude (loaded that way regardless of input bit
        // representation, via SFPLOAD instr_mod0) -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPMAD 2-cycle write latency; SFP_STOCH_RND reads LREG0 next.
        // SFPNOP (not TTI_NOP) so the bubble lands in the SFPU pipe and the
        // recorded body contains only SFPU-pipe opcodes.
        TTI_SFPNOP;
        // fp32 -> int sign-magnitude. LCONST_0 (LREG9) provides the 0.0 descale.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0 /*imm8*/,
            p_sfpu::LCONST_0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false>
void dequant_init(const uint zero_point) {
    // One-time setup for calculate_dequant; see quant_init for the
    // record/replay rationale. The caller passes -zero_point (so the
    // recorded body computes (A + LREG2) * B = (A - zero_point) * B).
    //
    // The recorded body is identical for both SIGN_MAGNITUDE_FORMAT variants;
    // the int32 input conversion happens in SFPLOAD's instr_mod0.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    lltt::record<lltt::NoExec>(DEQUANT_REPLAY_SLOT, DEQUANT_REPLAY_LEN);
    {
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        // SFPADD = VA*VB + VC ; with LCONST_1 (LREG10) = 1.0 this collapses
        // to A + LREG2 (= A + zero_point as loaded by the caller).
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPADD has a 2-cycle write latency on LREG0; SFPMUL reads it next.
        TTI_SFPNOP;
        // SFPMUL with LCONST_0 (LREG9 = 0.0) ignored as +C :
        // LREG0 = (A + LREG2) * LREG1.
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPMUL has a 2-cycle write latency on LREG0; the SFPSTORE that
        // follows the replay reads it. Keep this NOP inside the recorded
        // body rather than relying on the implicit gap between the replay
        // completing and TT_SFPSTORE issuing.
        TTI_SFPNOP;
    }
}

}  // namespace ckernel::sfpu
