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
// Int8 output records a longer body (the offset-128 pack) into the same family
// slot; slot spacing uses each family's max body length so the three ops can
// still coexist.
//
// Body content (see the inits for the exact emission order). SFPMAD / SFPADD have a 2-cycle
// write latency on WH, so each is followed by one SFPNOP before its result is read.
// <clamp + convert> is SFPSWAP (max lo), SFPSWAP (min hi), SFPIADD (see RNE_MAGIC_FP32).
//   QUANT   (5)            : SFPMAD, SFPNOP, <clamp + convert>                               (int32 / uint8 output)
//   QUANT   (int8-out, 6)  : SFPMAD, SFPNOP, <clamp + convert>, SFPXOR
//   REQUANT (8)            : SFPCAST(int->fp32), SFPMAD, SFPNOP, SFPADD, SFPNOP, <clamp + convert>
//   REQUANT (int8-out, 9)  : SFPCAST(int->fp32), SFPMAD, SFPNOP, SFPADD, SFPNOP, <clamp + convert>, SFPXOR
//   DEQUANT (5)            : SFPCAST(int->fp32), SFPADD, SFPNOP, SFPMUL, SFPNOP
//
// The int32 and uint8 outputs differ only in the clamp constants, so they share one body length.
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN = 5;
constexpr std::uint32_t QUANT_REPLAY_LEN_INT8_OUT = 6;
constexpr std::uint32_t QUANT_REPLAY_LEN_MAX = QUANT_REPLAY_LEN_INT8_OUT;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t REQUANT_REPLAY_LEN = 8;
constexpr std::uint32_t REQUANT_REPLAY_LEN_INT8_OUT = 9;
constexpr std::uint32_t REQUANT_REPLAY_LEN_MAX = REQUANT_REPLAY_LEN_INT8_OUT;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t DEQUANT_REPLAY_LEN = 5;

// Int8 L1 pack path:
// Int8 output is packed through the UInt8 packer, which passes the low byte of the dest word
// through unsigned (saturating to [0, 255]). So the SFPU leaves the 2's complement byte there as a
// value in [0, 255]: b = (n + 128) ^ 0x80, with n + 128 in [0, 255] after _rne_clamp_convert_.
// The byte is stored via INT32_2S_COMP and the UInt8 packer then emits raw b. (FP32_TO_INT8
// rounding cannot be used: its sign-magnitude result has a 7-bit magnitude, so it saturates at
// +/-127 and cannot represent -128.)
constexpr std::uint32_t INT8_SIGN_MASK = 0x00000080u;

// Round to nearest even, saturate and convert without STOCH_RND:
// STOCH_RND's non-stochastic mode rounds half-way cases away from zero on WH/BH (sfpi's
// SFPSTOCHRND_RND_EVEN is only a deprecated alias of SFPSTOCHRND_RND_NEAREST), and its FP32_TO_INT8
// mode cannot produce -128, while torch and ONNX quantization round ties to even into [-128, 127].
// The SFPU adder does round to nearest even, so adding RNE_MAGIC (1.5 * 2^23) to any |v| < 2^22 gives
// m = RNE(v) + RNE_MAGIC in [2^23, 2^24), where the fp32 spacing is 1 and the bits of m are
// bits(RNE_MAGIC) + RNE(v). So, with t = RNE_MAGIC - zero-point for quant (the zero point is applied
// after rounding, as torch does) and t = RNE_MAGIC for requant (whose host-folded zero point is part of v):
//   m = min(max(m, t + LO), t + HI)   two SFPSWAPs against LREG12 / LREG13, whose SFPSWAP writes are dropped
//   n = bits(m) - bits(t)             one SFPIADD; a 2's-complement int in [LO, HI]
// [LO, HI] is [-128, 127] for the int8 range and [0, 255] for uint8. Clamping m first also keeps it in
// that window for any |v|, including +/-inf. The int8 output adds 128 to n (excess-128) and XORs 0x80 to
// get its byte (see the Int8 L1 pack path above).
constexpr std::uint32_t RNE_MAGIC_FP32 = 0x4b400000u;      // 12582912.0f = 1.5 * 2^23
constexpr std::uint32_t RNE_NEG_MAGIC_BITS = 0xb4c00000u;  // -bits(RNE_MAGIC_FP32) as an int32

template <DataFormat OUTPUT_FORMAT>
constexpr int RNE_CLAMP_LO = OUTPUT_FORMAT == DataFormat::UInt8 ? 0 : -128;
template <DataFormat OUTPUT_FORMAT>
constexpr int RNE_CLAMP_HI = OUTPUT_FORMAT == DataFormat::UInt8 ? 255 : 127;

// From t in T_LREG: LREG12 = t + LO and LREG13 = t + HI (bits(t + k) = bits(t) + k inside the window),
// then T_LREG = -bits(t) (+128 for the int8 output). LREG0 is scratch.
template <DataFormat OUTPUT_FORMAT, std::uint32_t T_LREG>
inline void _rne_clamp_init_() {
    TTI_SFPIADD(
        RNE_CLAMP_LO<OUTPUT_FORMAT> & 0xfff,
        T_LREG,
        p_sfpu::LREG0,
        sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCONFIG(0, 12, 0);  // LREG12 = t + LO
    TTI_SFPIADD(
        RNE_CLAMP_HI<OUTPUT_FORMAT> & 0xfff,
        T_LREG,
        p_sfpu::LREG0,
        sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCONFIG(0, 13, 0);  // LREG13 = t + HI
    TTI_SFPIADD(0, p_sfpu::LCONST_0, T_LREG, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
        TTI_SFPIADD(128, T_LREG, T_LREG, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    }
}

// LREG0 holds m = RNE(v) + RNE_MAGIC; NEG_T_LREG holds -bits(t) (+128). Leaves n in LREG0.
template <std::uint32_t NEG_T_LREG>
inline void _rne_clamp_convert_() {
    TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);  // m = max(m, t + LO)
    TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // m = min(m, t + HI)
    TTI_SFPIADD(0, NEG_T_LREG, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
}

// Int8 L1 input path:
// Int8 input is read through the UInt8 unpacker, which zero-extends the raw byte
// into DST. The native Int8 unpacker is avoided because it widens negatives into a form that
// can be mishandled by subsequent SFPU INT32->FP32 cast.
//
// XOR-ing that byte with 0x80 recovers the excess-128 value e = s + 128, where s
// is the true signed value.
// For u < 128, u ^ 0x80 = u + 128 = s + 128
// For u >= 128 (s = u - 256), u ^ 0x80 = u - 128 = s + 128
// SFPU then casts e to fp32 and runs the normal formula. The constant 128 term is cancelled on the
// host by folding -128 into the zero point (dequant) or by an extra -128 * scale
// term in the zero-point constant (requant).
inline void _int8_input_unbias_() { TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0); }

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

    // The recorded body leaves a 2's-complement int32 (or a non-negative uint8) in LREG0, so the
    // default 2's-complement output is stored raw. The sign-magnitude variant runs it through the
    // store's sign-magnitude <-> 2's-complement swap, which is its own inverse.
    constexpr InstrModLoadStore out_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : InstrModLoadStore::INT32;

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
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN);                        // RNE MAD + pack
        TT_SFPSTORE(p_sfpu::LREG0, out_mode, ADDR_MOD_2, out_off);                // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void calculate_requant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input to requant (int32, sign-magnitude or 2's complement bits or UInt8-unpacked int8 byte).
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

    constexpr InstrModLoadStore in_mode =
        (SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;
    // The recorded body leaves a 2's-complement int32 (or a non-negative uint8) in LREG0; see
    // calculate_quant_int32 for the output store mode.
    constexpr InstrModLoadStore out_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32_2S_COMP : InstrModLoadStore::INT32;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute
    // (the int->fp cast doesn't touch LREG1 so reordering is safe), replay
    // the recorded body, then SFPSTORE under ADDR_MOD_2 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, in_mode, ADDR_MOD_3, in0_off);  // operand A (int32 -> sign-magn LREG0)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80 (excess-128)
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN);      // CAST + RNE MAD + pack
        TT_SFPSTORE(p_sfpu::LREG0, out_mode, ADDR_MOD_2, out_off);  // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_quant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output:
    // The RNE MAD + offset-128 pack body is recorded once into QUANT_REPLAY_SLOT and replayed,
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_3, in0_off);  // operand A (fp32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN_INT8_OUT);               // RNE MAD + offset-128 pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_2, out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool INT8_INPUT = false>
inline void calculate_requant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output:
    // The CAST + RNE MAD + offset-128 pack body is recorded once into REQUANT_REPLAY_SLOT and replayed.
    // The int8-input unbias (byte ^ 0x80) stays inline before the replay.
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, in0_off);  // operand A (int32/byte)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);  // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN_INT8_OUT);  // CAST + RNE MAD + offset-128 pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_2, out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void calculate_dequant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A[LREG0] is input to dequant (int32, sign-magnitude or 2's complement bits or UInt8-unpacked int8 byte).
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
        (SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

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
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80
        }
        lltt::replay(DEQUANT_REPLAY_SLOT, DEQUANT_REPLAY_LEN);                     // CAST + ADD + SFPNOP + MUL + SFPNOP
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_2, out_off);  // store fp32 + dst_reg += 2
    }
}

// OUTPUT_FORMAT selects the quantized output tensor dtype: Int32 (the default), UInt8, or Int8. All
// three round to nearest even and saturate through _rne_clamp_convert_: Int32 holds int8-range values
// [-128, 127] in an int32 container, UInt8 [0, 255], and Int8 packs the excess-128 byte.
template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32>
void quant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "quant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    // LREG5 = RNE_MAGIC (the MAD addend); LREG2 = t = RNE_MAGIC - zero-point, exact for an integer zero
    // point, which _rne_clamp_init_ turns into the clamp bounds and then -bits(t) (+128).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG5, p_sfpu::LREG2, 0 /*mod1*/);
    TTI_SFPNOP;
    _rne_clamp_init_<OUTPUT_FORMAT, p_sfpu::LREG2>();
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
    _quant_kernels_configure_dest_incr_addrmod_();

    // The replay buffer feeds the SFPU pipe directly, so the recorded bodies use TTI_SFPNOP (SFPU NOP)
    // rather than the generic Tensix TTI_NOP for their pipeline bubbles.
    constexpr std::uint32_t REPLAY_LEN =
        OUTPUT_FORMAT == DataFormat::Int8 ? QUANT_REPLAY_LEN_INT8_OUT : QUANT_REPLAY_LEN;
    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        // m = RNE(A * B) + RNE_MAGIC: the single rounding step of the MAD rounds to nearest even
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /*mod1*/);
        TTI_SFPNOP;
        _rne_clamp_convert_<p_sfpu::LREG2>();  // n = clamp(RNE(A * B) + zp) (+128)
        if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
            TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // b = (n + 128) ^ 0x80
        }
    }
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32,
    bool INT8_INPUT = false>
void requant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "requant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    // The body rounds the whole expression q * (s_in / s_out) + zp to nearest even, zp being the
    // host-folded z_out - z_in * (s_in / s_out) in LREG2; this matches the op's golden
    // round((q - z_in) * (s_in / s_out) + z_out). With t = RNE_MAGIC, LREG6 ends up as -bits(t) (+128).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
    _sfpu_load_imm32_(p_sfpu::LREG6, RNE_MAGIC_FP32);
    _rne_clamp_init_<OUTPUT_FORMAT, p_sfpu::LREG6>();
    if constexpr (INT8_INPUT || OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN =
        OUTPUT_FORMAT == DataFormat::Int8 ? REQUANT_REPLAY_LEN_INT8_OUT : REQUANT_REPLAY_LEN;
    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        // int32 sign-magnitude (loaded that way regardless of input bit
        // representation, via SFPLOAD instr_mod0) -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);  // v = A * B + zp
        TTI_SFPNOP;
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /*mod1*/);  // m = RNE(v) + MAGIC
        TTI_SFPNOP;
        _rne_clamp_convert_<p_sfpu::LREG6>();  // n = clamp(RNE(v)) (+128)
        if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
            TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // b = (n + 128) ^ 0x80
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false, bool INT8_INPUT = false>
void dequant_init(const uint zero_point) {
    // One-time setup for calculate_dequant; see quant_init for the
    // record/replay rationale. The caller passes -zero_point (so the
    // recorded body computes (A + LREG2) * B = (A - zero_point) * B).
    //
    // The recorded body is identical for both SIGN_MAGNITUDE_FORMAT variants;
    // the int32 input conversion happens in SFPLOAD's instr_mod0.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (INT8_INPUT) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
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
