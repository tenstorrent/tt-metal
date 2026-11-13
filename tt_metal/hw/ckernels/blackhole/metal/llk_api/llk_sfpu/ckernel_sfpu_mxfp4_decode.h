// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SFPU decoders for MXFP4 stored as relabelled bfloat4_b: bits += 3 * (bits & 0x00200000) on each FP32
// value in Dest. In units of half the MXFP4 scale, e2m1 magnitudes are {0, 1, 2, 3, 4, 6, 8, 12}; the host
// stores 8 as 5 and 12 as 7 so every value is a bfloat4_b code, and this decode maps them back: a value
// with the mantissa's 0.25 bit set (1.25 * 2^e, 1.75 * 2^e) drops that bit and gains one exponent step.
//
// Copy of tt-blaze's blaze/ops/mxfp4_remap_streaming_matmul/kernels/mxfp4_decode.hpp (the decoders its
// Kimi K3 MoE decode runs); tt_metal/tt-llk/tests/sources/mxfp4_carry_decode_test.cpp checks them
// bit-exact against MXFP4 for every e8m0 scale and times them.
//
// Thread use: mxfp4_sfpu_configure() programs SFPU state shared by the threads (macro templates and
// sequences, LReg constants), once; mxfp4_sfpu_thread_init() runs on every thread (MATH or PACK) that
// decodes; mxfp4_sfpu_tile() decodes one tile in Dest.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {
// Remapped MXFP4 -> MXFP4: a value with the mantissa's 0.25 bit set (1.25 * 2^e, 1.75 * 2^e) drops that bit
// and gains one exponent step (2^(e+1), 1.5 * 2^(e+1)). LRegs hold FP32 layout: the bit is 0x00200000.
template <int ITERATIONS>
inline void mxfp4_remap() {
    // Mask kept in its own LReg; immediate DEST offsets instead of a pointer increment per vector:
    // load, and, add, shift, add, store = 6 instructions per 32 values.
    const sfpi::vInt mask = 0x00200000;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt b = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[d]));
        sfpi::vInt m = b & mask;
        b = b + m + (m << 1);
        sfpi::dst_reg[d] = sfpi::as<sfpi::vFloat>(b);
    }
    static_assert(ITERATIONS == 8);
    // INCRWC takes at most 7 rows: 6 + 6 + 4 rows = one face.
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 3;
    sfpi::dst_reg += 2;
}

// SFPLOADMACRO variant of the same decode, over a whole tile (after _llk_math_eltwise_sfpu_start_). Per
// vector v, software issues three instructions and the macro schedules two:
//   slot 3v    SFPLOADMACRO R = Dst[v]   (R alternates L0 / L4; macro v & 1)
//   slot 3v+1  SFPAND   T = R & 0x00200000                       simple sub-unit
//   slot 3v+2  SFPMUL24 T3[v & 1] = T * 3                         MAD sub-unit (low 23 bits)
//   slot 3v+5  (scheduled) SFPIADD L16 = R + T3[v & 1]           simple sub-unit, delay 4
//   slot 3v+6  (scheduled) SFPSTORE Dst[v] = L16                 store sub-unit, delay 5
// The scheduled IADD lands on a MUL24 slot and the scheduled store on a load slot, so neither displaces a
// software instruction. Delays count issued SFPU instructions. R is read for the last time at 3v+5, before
// the macro two vectors later reloads it at 3v+6. Silicon (Blackhole LLK test): 107 cycles per tile,
// against 230 for mxfp4_remap<8> over four faces; bit-exact in both Dest modes.
constexpr uint32_t mxfp4_macro_mask_reg = 3;
constexpr uint32_t mxfp4_macro_three_reg = 7;
constexpr uint32_t mxfp4_macro_t_reg = 1;
constexpr uint32_t mxfp4_macro_t3_reg0 = 2;
constexpr uint32_t mxfp4_macro_t3_reg1 = 6;

// Configure the two load macros (nothing else in the op may use SFPLOADMACRO), and the constants.
inline void mxfp4_macro_configure() {
    constexpr uint32_t iadd_reg_reg = 4;  // SFPIADD: VD = VC + VD, condition codes untouched
    TTI_SFPCONFIG(0, 0xF, 1);             // LaneConfig defaults: backdoor template loads enabled
    TTI_SFPNOP;
    // An instruction with VD = 12 + i is stored as InstructionTemplate[i] instead of executing.
    TTI_SFPIADD(0, mxfp4_macro_t3_reg0, 12, iadd_reg_reg);
    TTI_SFPNOP;
    TTI_SFPIADD(0, mxfp4_macro_t3_reg1, 13, iadd_reg_reg);
    TTI_SFPNOP;
    // Sequence bytes: simple unit in bits 0..7, store unit in bits 24..31. Bit 7 = take VB from the macro's
    // register, bit 6 = write (simple) / read (store) LReg16, bits 5..3 = delay, bits 2..0 = 4 + template
    // index (simple) or 3 = SFPSTORE (store).
    constexpr uint32_t store_byte = 0x40 | (5 << 3) | 3;
    constexpr uint32_t seq0 = (store_byte << 24) | (0x80 | 0x40 | (4 << 3) | 4);
    constexpr uint32_t seq1 = (store_byte << 24) | (0x80 | 0x40 | (4 << 3) | 5);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq0 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq0 >> 16);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq1 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq1 >> 16);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPNOP;
    // Misc: StoreMod0 = 0; macros 0 and 1 store with their load's Mod0 (bits 4, 5); the simple and store
    // sub-units count delays in issued instructions (bits 8, 11).
    TTI_SFPCONFIG(0x930, 8, 1);
    TTI_SFPNOP;
    TTI_SFPLOADI(mxfp4_macro_mask_reg, 0, 0x0020);  // BF16 immediate (upper half): 0x00200000
    TTI_SFPLOADI(mxfp4_macro_three_reg, sfpi::SFPLOADI_MOD0_LOWER, 3);
    TTI_SFPLOADI(mxfp4_macro_three_reg, sfpi::SFPLOADI_MOD0_UPPER, 0);
}

inline void mxfp4_macro_decode_tile() {
    constexpr uint32_t fmt_implied = 0;
    constexpr uint32_t addr_no_incr = 7;
    constexpr uint32_t and_use_vb = 1;
    constexpr uint32_t mul24_lower = 0;
    constexpr uint32_t zero_reg = 9;  // constant 0: keeps SFPMUL24's VC exponent zero
    constexpr uint32_t chain[2] = {0, 4};
    constexpr uint32_t t3[2] = {mxfp4_macro_t3_reg0, mxfp4_macro_t3_reg1};
#pragma GCC unroll 32
    for (uint32_t v = 0; v < 32; ++v) {
        const uint32_t r = chain[v & 1];
        // The address's low bit is unused by the load; it carries bit 2 of the register index.
        TTI_SFPLOADMACRO(((v & 1) << 2) | (r & 3), fmt_implied, addr_no_incr, v * 2 + (r >> 2));
        TTI_SFPAND(r, mxfp4_macro_mask_reg, mxfp4_macro_t_reg, and_use_vb);
        TTI_SFPMUL24(mxfp4_macro_t_reg, mxfp4_macro_three_reg, zero_reg, t3[v & 1], mul24_lower);
    }
    // Let the last two vectors' scheduled IADD and store count down and run.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Two-load SFPLOADMACRO variant: two issued instructions per vector, both SFPLOADMACROs of the same Dst row,
// and the macros schedule the rest:
//   slot 2v    SFPLOADMACRO A[v & 1] = Dst[v]    (macro 0)
//   slot 2v+1  (scheduled) SFPAND   A = A & 0x00200000               simple sub-unit, delay 0
//   slot 2v+1  SFPLOADMACRO B[v & 1] = Dst[v]    (macro 1 + (v & 1))
//   slot 2v+2  (scheduled) SFPMUL24 A = A * 3                        MAD sub-unit, delay 1
//   slot 2v+4  (scheduled) SFPIADD  L16 = B + A[v & 1]               simple sub-unit, delay 2
//   slot 2v+5  (scheduled) SFPSTORE Dst[v] = L16                     store sub-unit, delay 3
// ANDs land on odd slots and IADDs on even ones. The IADD names A[v & 1] through its template, so A rotates
// over two registers only, and vector v + 2 reloads A[v & 1] on slot 2v+4, the slot on which the IADD of
// vector v reads it: a scheduled instruction reads its LRegs before a same-cycle SFPLOADMACRO writes them
// (not documented; silicon bit-exact). Blackhole LLK test: 74 cycles per tile against 107 for the 3-issue one.
constexpr uint32_t mxfp4_macro2_a[2] = {0, 4};
constexpr uint32_t mxfp4_macro2_b[2] = {1, 5};

inline void mxfp4_macro2_configure() {
    constexpr uint32_t iadd_reg_reg = 4;
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SFPNOP;
    TTI_SFPAND(0, mxfp4_macro_mask_reg, 12, 1);  // template 0: A &= mask (VB from the macro)
    TTI_SFPNOP;
    TTI_SFPMUL24(mxfp4_macro_three_reg, 0, 9, 13, 0);  // template 1: A = 3 * A (low 23 bits)
    TTI_SFPNOP;
    TTI_SFPIADD(0, mxfp4_macro2_a[0], 14, iadd_reg_reg);  // templates 2, 3: L16 = B + A[i]
    TTI_SFPNOP;
    TTI_SFPIADD(0, mxfp4_macro2_a[1], 15, iadd_reg_reg);
    TTI_SFPNOP;
    constexpr uint32_t seq_a = ((0x80 | (1 << 3) | 5) << 8) | (0x80 | (0 << 3) | 4);
    constexpr uint32_t store = 0x40 | (3 << 3) | 3;
    constexpr uint32_t seq_b0 = (store << 24) | (0x80 | 0x40 | (2 << 3) | 6);
    constexpr uint32_t seq_b1 = (store << 24) | (0x80 | 0x40 | (2 << 3) | 7);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq_a & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq_a >> 16);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq_b0 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq_b0 >> 16);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq_b1 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq_b1 >> 16);
    TTI_SFPCONFIG(0, 4 + 2, 0);
    TTI_SFPNOP;
    // Misc: StoreMod0 = 0; macros 1 and 2 store with their load's Mod0; all sub-units count issued instructions.
    TTI_SFPCONFIG(0xF60, 8, 1);
    TTI_SFPNOP;
    TTI_SFPLOADI(mxfp4_macro_mask_reg, 0, 0x0020);
    TTI_SFPLOADI(mxfp4_macro_three_reg, sfpi::SFPLOADI_MOD0_LOWER, 3);
    TTI_SFPLOADI(mxfp4_macro_three_reg, sfpi::SFPLOADI_MOD0_UPPER, 0);
}

inline void mxfp4_macro2_decode_tile() {
    constexpr uint32_t fmt_implied = 0;
    constexpr uint32_t addr_no_incr = 7;
#pragma GCC unroll 32
    for (uint32_t v = 0; v < 32; ++v) {
        const uint32_t a = mxfp4_macro2_a[v & 1];
        const uint32_t b = mxfp4_macro2_b[v & 1];
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), fmt_implied, addr_no_incr, v * 2 + (a >> 2));
        TTI_SFPLOADMACRO(((1 + (v & 1)) << 2) | (b & 3), fmt_implied, addr_no_incr, v * 2 + (b >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Replay variant of the two-load decoder (sfpu_macro = 3): the loads address Dst through the SFPU RWC instead
// of immediates, so one 16-instruction recording (8 vectors) replays four times per tile, and TRISC issues
// 4 REPLAYs + 5 NOPs per tile instead of 69 instructions. The B load advances the RWC by one vector (2 rows)
// through ADDR_MOD_6; A loads use ADDR_MOD_7 (no increment). Replay slots 0..15 are the SFPU half of the
// per-thread buffer (ckernel::math::replay_buf_offset = 16 splits it with the FPU).
constexpr uint32_t mxfp4_replay_start = 0;
constexpr uint32_t mxfp4_replay_len = 16;

inline void mxfp4_macro2_replay_thread_init() {
    ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    ckernel::load_replay_buf<ckernel::NoExec>(mxfp4_replay_start, mxfp4_replay_len, [] {
#pragma GCC unroll 4
        for (uint32_t u = 0; u < 4; ++u) {
            // A0 = L0, B0 = L1, A1 = L4, B1 = L5: VDLo in the register field, VDHi in the address's low bit.
            TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 0);
            TTI_SFPLOADMACRO((1 << 2) | 1, 0, 6, 0);
            TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, 1);
            TTI_SFPLOADMACRO((2 << 2) | 1, 0, 6, 1);
        }
    });
}

inline void mxfp4_macro2_replay_decode_tile() {
    lltt::replay(mxfp4_replay_start, mxfp4_replay_len);
    lltt::replay(mxfp4_replay_start, mxfp4_replay_len);
    lltt::replay(mxfp4_replay_start, mxfp4_replay_len);
    lltt::replay(mxfp4_replay_start, mxfp4_replay_len);
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Decoder dispatch. Macro: 0 sfpi loop, 1 three-issue SFPLOADMACRO, 2 two-load SFPLOADMACRO, 3 = 2 replayed.
template <uint32_t Macro>
inline void mxfp4_sfpu_configure() {  // shared SFPU state: run on one thread
    if constexpr (Macro == 1) {
        mxfp4_macro_configure();
    } else if constexpr (Macro == 2 || Macro == 3) {
        mxfp4_macro2_configure();
    }
}

template <uint32_t Macro>
inline void mxfp4_sfpu_thread_init() {  // per-thread state: run on every thread that decodes
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, false>();
    if constexpr (Macro == 3) {
        mxfp4_macro2_replay_thread_init();
    }
}

// Reload the decoder's LReg constants (mask, 3) after another SFPU user may have clobbered them.
template <uint32_t Macro>
inline void mxfp4_sfpu_reload_constants() {
    if constexpr (Macro != 0) {
        TTI_SFPLOADI(mxfp4_macro_mask_reg, 0, 0x0020);
        TTI_SFPLOADI(mxfp4_macro_three_reg, sfpi::SFPLOADI_MOD0_LOWER, 3);
        TTI_SFPLOADI(mxfp4_macro_three_reg, sfpi::SFPLOADI_MOD0_UPPER, 0);
    }
}

template <uint32_t Macro>
inline void mxfp4_sfpu_tile(uint32_t t) {
    if constexpr (Macro == 0) {
        _llk_math_eltwise_unary_sfpu_params_(mxfp4_remap<8>, t, VectorMode::RC);
    } else {
        _llk_math_eltwise_sfpu_start_(t);
        if constexpr (Macro == 1) {
            mxfp4_macro_decode_tile();
        } else if constexpr (Macro == 2) {
            mxfp4_macro2_decode_tile();
        } else {
            mxfp4_macro2_replay_decode_tile();
        }
        _llk_math_eltwise_sfpu_done_();
    }
}
}  // namespace ckernel::sfpu
