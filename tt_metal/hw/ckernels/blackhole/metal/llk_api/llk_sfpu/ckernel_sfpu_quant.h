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
// op's {quant,requant,dequant}_init and then replayed by every
// invocation of the matching _{quant,requant,dequant}_int32_ kernel. The
// two SIGN_MAGNITUDE_FORMAT variants of a given kernel safely share its
// slot - the init writes whichever variant it was templated for and the
// kernel uses the matching REPLAY_LEN. Distinct slots between kernels are
// required so a single compute kernel can mix all three ops without each
// init clobbering the others' recordings.
//
// Body content (see the inits for the exact emission order). No SFPNOPs are
// emitted: on Blackhole the SFPU implicitly stalls on read-after-write
// hazards between back-to-back fp32 ops, so SFPMAD->STOCH_RND,
// SFPADD->SFPMUL and SFPMUL->SFPSTORE don't need explicit pipeline bubbles.
//   QUANT   ( 2s-comp, 4 ) : SFPMAD, STOCH_RND, SFPCAST, SFPSETSGN
//   QUANT   (sign-magn, 2) : SFPMAD, STOCH_RND
//   REQUANT ( 2s-comp, 7 ) : SFPCAST+SFPSETSGN(in), SFPCAST(int->fp32), SFPMAD,
//                             STOCH_RND, SFPCAST+SFPSETSGN(out)
//   REQUANT (sign-magn, 3) : SFPCAST(int->fp32), SFPMAD, STOCH_RND
//   DEQUANT ( 2s-comp, 5 ) : SFPCAST+SFPSETSGN(in), SFPCAST(int->fp32),
//                             SFPADD, SFPMUL
//   DEQUANT (sign-magn, 3) : SFPCAST(int->fp32), SFPADD, SFPMUL
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN_2S_COMP = 4;
constexpr std::uint32_t QUANT_REPLAY_LEN_SIGN_MAGN = 2;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t REQUANT_REPLAY_LEN_2S_COMP = 7;
constexpr std::uint32_t REQUANT_REPLAY_LEN_SIGN_MAGN = 3;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_2S_COMP = 5;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_SIGN_MAGN = 3;

// Direction-neutral alias for the SFPCAST+SFPSETSGN combo emitted by
// apply_sign_magnitude_conversion. The combo is the Blackhole workaround
// for the SFPCAST RTL bug (tenstorrent/tt-llk-bh#16) and is symmetric:
// it swaps between int32 sign-magnitude and 2's-complement representations
// regardless of which side is the source. The named enum value
// InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP describes only one of those
// directions; tt-llk convention (matching the canonical bug-fix commit and
// the other int32 SFPU kernels) is to use it for both directions. The
// sibling value INT32_2S_COMP_TO_INT_SIGN_MAGN has a known HW bug
// (sign-mag -0 -> mostneg int32) and is not used in any tt-llk kernel.
constexpr auto INT_REPR_SWAP_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

// Configure ADDR_MOD_6 with dest auto-increment of one SFPU dst row
// (sfpi::SFP_DESTREG_STRIDE == 2 dst-address units) so the per-iteration
// SFPSTORE walks dst_reg through the face's 4-row x 8-col blocks. Replaces
// sfpi::dst_reg++ in the kernel bodies and lets each loop be purely TTI-issued.
// Called once by each _init_{quant,requant,dequant}_int32_ since quant_int32
// isn't in the LLK init's "configure ADDR_MOD_6 with dest+=2" allow-list.
inline void _quant_kernels_configure_dest_incr_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = sfpi::SFP_DESTREG_STRIDE},
    }
        .set(ADDR_MOD_6);
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
void quant_init(const uint zero_point) {
    // One-time setup for calculate_quant_int32:
    //   1. load the fp32 zero-point constant into LREG2;
    //   2. program ADDR_MOD_6 with dest+=2 for the per-iteration SFPSTORE;
    //   3. record the register-only compute body into the SFPU replay buffer
    //      under QUANT_REPLAY_SLOT (NoExec - we don't want to issue SFPMAD/
    //      STOCH_RND against undefined LREG0/LREG1 contents at record time).
    // Subsequent _quant_int32_ calls replay the recorded body, shrinking the
    // unrolled binary from ~ITERATIONS*REPLAY_LEN body instructions down to
    // one replay invocation per iteration.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point). The Blackhole SFPU
        // implicitly stalls SFP_STOCH_RND below until SFPMAD's LREG0 write
        // retires, so no explicit pipeline-bubble TTI_SFPNOP is needed here.
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // fp32 -> int sign-magnitude. LCONST_0 (LREG9) is the HW-provided 0.0
        // used as the zero descale.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0 /*imm8*/,
            p_sfpu::LCONST_0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            // STOCH_RND output above is in sign-magnitude form; convert to
            // 2's-complement so the trailing INT32_2S_COMP SFPSTORE writes
            // out 2's-complement bits (on BH the store mode itself is a
            // no-op, so the bits in LREG0 are what land in memory). LREG4
            // is the helper's scratch destination; LREG0 receives the
            // sign-fixed result. See INT_REPR_SWAP_CAST above for why the
            // cast mode constant is direction-neutral.
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG4, INT_REPR_SWAP_CAST);
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
void requant_init(const uint zero_point) {
    // One-time setup for requant_int32; see quant_init for the
    // record/replay rationale. Loads the zero point into LREG2, programs
    // ADDR_MOD_6 with dest+=2, then records the register-only compute into
    // REQUANT_REPLAY_SLOT (NoExec).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            // Input arrives in 2's-complement bits in LREG0 (the upstream
            // quant kernel stores 2's-complement when SIGN_MAGNITUDE_FORMAT
            // is false, and the INT32_2S_COMP SFPLOAD mode is a no-op on
            // BH). Convert to sign-magnitude so the int->fp32 SFPCAST below
            // sees its expected input. Uses the same cast+SETSGN combo as
            // the output side (see INT_REPR_SWAP_CAST above for why this
            // works in both directions).
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG4, INT_REPR_SWAP_CAST);
        }
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point). BH SFPU implicitly
        // stalls STOCH_RND below until SFPMAD's LREG0 write retires, so no
        // explicit pipeline-bubble TTI_SFPNOP is needed here.
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // fp32 -> int sign-magnitude. LCONST_0 (LREG9) provides the 0.0 descale.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0 /*imm8*/,
            p_sfpu::LCONST_0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            // STOCH_RND output is sign-magnitude; convert to 2's-complement
            // for the trailing INT32_2S_COMP SFPSTORE. Same cast+SETSGN
            // combo as the input side; see INT_REPR_SWAP_CAST above.
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG4, INT_REPR_SWAP_CAST);
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
void dequant_init(const uint zero_point) {
    // One-time setup for calculate_dequant_int32; see quant_init for the
    // record/replay rationale. The caller passes -zero_point (so the
    // recorded body computes (A + LREG2) * B = (A - zero_point) * B).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(DEQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        if constexpr (!SIGN_MAGNITUDE_FORMAT) {
            // Input arrives in 2's-complement bits in LREG0 (INT32_2S_COMP
            // SFPLOAD is a no-op on BH); convert to sign-magnitude so the
            // int->fp32 SFPCAST below sees its expected input. Same
            // cast+SETSGN combo as the other sites; see INT_REPR_SWAP_CAST
            // above.
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG4, INT_REPR_SWAP_CAST);
        }
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        // SFPADD = VA*VB + VC ; with LCONST_1 (LREG10) = 1.0 this collapses
        // to A + LREG2 (= A + zero_point as loaded by the caller). BH SFPU
        // implicitly stalls SFPMUL below until SFPADD's LREG0 write retires.
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // SFPMUL with LCONST_0 (LREG9 = 0.0) ignored as +C :
        // LREG0 = (A + LREG2) * LREG1. The TT_SFPSTORE outside the replay
        // (which reads LREG0) is similarly handled by the SFPU's implicit
        // RAW stall, so no trailing TTI_SFPNOP is required.
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0 /*mod1*/);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_quant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input (fp32).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_quant_int32_.
    // Output is int32 scaled to int8 range (sign-magnitude or 2's-complement).
    //
    // Tile layout in Dest: each tile occupies 64 dest-address units (4 faces
    // x 16 addr/face). Each SFPLOAD/SFPSTORE moves 4 dest rows x 8 SFPU lanes,
    // so advancing dst_reg by +2 between iterations walks one face's eight
    // 4-row x 8-col blocks (= one full call site, ITERATIONS == 8).
    //
    // The replay-buffer body at QUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_quant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: inline TT_SFPLOADs (variable addresses can't live inside
    // the replay buffer because TT_* macros write to instrn_buffer[0]), replay
    // the recorded compute, then SFPSTORE under ADDR_MOD_6 which also auto-
    // advances dst_reg by 2 for the next iteration's loads.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, in0_off);  // operand A (fp32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);  // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, REPLAY_LEN);                              // MAD + STOCH_RND + (CAST + SETSGN)
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);  // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_requant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input to requant (int32, sign-magnitude or 2's complement bits).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_requant_int32_.
    // Output is int32 scaled to int8 range (sign-magnitude or 2's-complement).
    //
    // The replay-buffer body at REQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_requant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute
    // (the input cast doesn't touch LREG1 so reordering is safe), replay the
    // recorded body, then SFPSTORE under ADDR_MOD_6 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off);  // operand A (int32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);           // operand B (fp32 scaler)
        lltt::replay(REQUANT_REPLAY_SLOT, REPLAY_LEN);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);  // store + dst_reg += 2
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
    // The replay-buffer body at DEQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_dequant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN =
        SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute,
    // replay the body, then SFPSTORE under ADDR_MOD_6 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off);  // operand A (int32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);           // operand B (fp32 scaler)
        lltt::replay(DEQUANT_REPLAY_SLOT, REPLAY_LEN);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, out_off);  // store fp32 + dst_reg += 2
    }
}

}  // namespace ckernel::sfpu
