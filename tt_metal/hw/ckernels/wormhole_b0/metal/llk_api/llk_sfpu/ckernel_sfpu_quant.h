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

// Replay-buffer slots for the register-only compute bodies of the quant /
// requant / dequant kernels. The body is recorded once by each op's
// _init_{quant,requant,dequant}_int32_ and then replayed by every invocation
// of the matching _{quant,requant,dequant}_int32_ kernel.
// Unlike the Blackhole port, the recorded body does not depend on
// SIGN_MAGNITUDE_FORMAT: WH performs the int32 sign-magnitude <-> 2's
// complement conversion in the SFPLOAD/SFPSTORE instr_mod0 field
// (INT32 = 4 vs INT32_2S_COMP = 12), and those load/store instructions are
// outside the replay window. So one slot per kernel suffices.
//
// Each body processes TWO dst rows per replay: row 0 lives in LREG0 (A) /
// LREG1 (B), row 1 in LREG3 (A) / LREG5 (B); LREG2 (zero point) and LREG4
// (int8 sign mask) are shared. The two rows' instruction chains are
// interleaved so that every 2-cycle producer (SFPMAD/SFPADD/SFPMUL) is
// followed by the other row's independent instruction instead of an SFPNOP:
// each consumer still issues >= 2 slots after its producer, exactly the
// distance the former MAD, SFPNOP, consumer sequence gave, and the per-element
// instruction sequence and operand order are unchanged.
//
// Distinct slots between kernels are required so a single compute kernel
// can mix all three ops without each init clobbering the others' recordings.
// Int8 output records a longer body (the offset-128 pack) into the same family
// slot; slot spacing uses each family's max body length so the three ops can
// still coexist. The three families together use all 32 replay entries.
//
// Body content (see the inits for the exact emission order; r0/r1 = row 0/1):
//   QUANT   (4)            : SFPMAD r0, SFPMAD r1, STOCH_RND r0, STOCH_RND r1
//   QUANT   (int8-out, 12) : SFPMAD r0, SFPMAD r1, <10-instr offset-128 pack:
//                            SFPSETCC r0, SFPMOV r0, SFPENCC, SFPSETCC r1, SFPMOV r1, SFPENCC,
//                            STOCH_RND r0, STOCH_RND r1, SFPXOR r0, SFPXOR r1>
//   REQUANT (6)            : SFPCAST r0, SFPCAST r1, SFPMAD r0, SFPMAD r1, STOCH_RND r0, STOCH_RND r1
//   REQUANT (int8-out, 14) : SFPCAST r0, SFPCAST r1, SFPMAD r0, SFPMAD r1, <10-instr pack>
//   DEQUANT (6)            : SFPCAST r0, SFPCAST r1, SFPADD r0, SFPADD r1, SFPMUL r0, SFPMUL r1
//
// The int8-out bodies fold the +128 offset into the fp32 zero-point once at init,
// so the per-iteration MAD already yields v + 128 and the pack needs no per-element SFPADDI.
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN = 4;
constexpr std::uint32_t QUANT_REPLAY_LEN_INT8_OUT = 12;
constexpr std::uint32_t QUANT_REPLAY_LEN_MAX = QUANT_REPLAY_LEN_INT8_OUT;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t REQUANT_REPLAY_LEN = 6;
constexpr std::uint32_t REQUANT_REPLAY_LEN_INT8_OUT = 14;
constexpr std::uint32_t REQUANT_REPLAY_LEN_MAX = REQUANT_REPLAY_LEN_INT8_OUT;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t DEQUANT_REPLAY_LEN = 6;

static_assert(DEQUANT_REPLAY_SLOT + DEQUANT_REPLAY_LEN <= 32, "quant replay bodies exceed the 32-entry replay buffer");

// Row-pair register plan (see the replay-body comment above).
constexpr std::uint32_t QUANT_ROW0_A = p_sfpu::LREG0;
constexpr std::uint32_t QUANT_ROW0_B = p_sfpu::LREG1;
constexpr std::uint32_t QUANT_ROW1_A = p_sfpu::LREG3;
constexpr std::uint32_t QUANT_ROW1_B = p_sfpu::LREG5;

// Int8 L1 pack path:
// FP32_TO_INT8 rounding cannot be used here since it produces a sign-magnitude result
// whose magnitude is only 7 bits, so it saturates at +/-127 and cannot represent -128
// (both -127 and -128 collapse to 127). It also leaves negatives in sign-magnitude form.
//
// Int8 output is instead packed through the UInt8 packer, which passes the low byte of
// the dest word through unsigned (saturating to [0, 255]). The SFPU keeps the 2's complement
// byte in the low 8 bits using an offset-128 that spans the full [-128, 127] range and saturates:
//   u = FP32_TO_UINT8(v + 128) (u in [0, 255], saturates both ends)
//   b = u ^ 0x80  (excess-128 -> 2's complement byte)
// The byte is stored via INT32_2S_COMP and the UInt8 packer then emits raw b.
constexpr std::uint32_t INT8_OFFSET_128_IMM16 = 0x4300u;
constexpr std::uint32_t INT8_SIGN_MASK = 0x00000080u;

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
// Applied to both rows of the pair.
inline void _int8_input_unbias_() {
    TTI_SFPXOR(0, p_sfpu::LREG4, QUANT_ROW0_A, 0);
    TTI_SFPXOR(0, p_sfpu::LREG4, QUANT_ROW1_A, 0);
}

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

// Load operands A and B of two consecutive dst rows: row 0 at the current
// dst_reg, row 1 one SFPU row (sfpi::SFP_DESTREG_STRIDE) further. The
// immediate is added to the Dst RWC, so neither load moves dst_reg
// (ADDR_MOD_3 = no increment). Variable addresses can't live inside the
// replay buffer because TT_* macros write to instrn_buffer[0].
template <InstrModLoadStore A_MODE>
inline void _quant_load_row_pair_(const std::uint32_t in0_off, const std::uint32_t in1_off) {
    TT_SFPLOAD(QUANT_ROW0_A, A_MODE, ADDR_MOD_3, in0_off);
    TT_SFPLOAD(QUANT_ROW0_B, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off);
    TT_SFPLOAD(QUANT_ROW1_A, A_MODE, ADDR_MOD_3, in0_off + sfpi::SFP_DESTREG_STRIDE);
    TT_SFPLOAD(QUANT_ROW1_B, InstrModLoadStore::FP32, ADDR_MOD_3, in1_off + sfpi::SFP_DESTREG_STRIDE);
}

// Store both rows under ADDR_MOD_2 (real slot 6): each store auto-advances
// dst_reg by one SFPU row, so the pair leaves dst_reg two rows further for
// the next pair's loads.
template <InstrModLoadStore OUT_MODE>
inline void _quant_store_row_pair_(const std::uint32_t out_off) {
    TT_SFPSTORE(QUANT_ROW0_A, OUT_MODE, ADDR_MOD_2, out_off);  // store row 0 + dst_reg += 2
    TT_SFPSTORE(QUANT_ROW1_A, OUT_MODE, ADDR_MOD_2, out_off);  // store row 1 + dst_reg += 2
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
    // by +2 between rows walks one face's eight 4-row x 8-col blocks
    // (= one full call site, ITERATIONS == 8). Rows are processed in pairs.
    //
    // The replay-buffer body at QUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_quant_int32_, which must run before the
    // first call here.
    static_assert(ITERATIONS % 2 == 0, "quant kernels process dst rows in pairs");
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore out_mode =
        SIGN_MAGNITUDE_FORMAT ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per row pair: inline TT_SFPLOADs, replay the recorded two-row compute,
    // then two SFPSTOREs under ADDR_MOD_2 which also advance dst_reg for the
    // next pair's loads.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        _quant_load_row_pair_<InstrModLoadStore::FP32>(in0_off, in1_off);  // A (fp32), B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN);                 // MAD r0/r1 + STOCH_RND r0/r1
        _quant_store_row_pair_<out_mode>(out_off);
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
    static_assert(ITERATIONS % 2 == 0, "quant kernels process dst rows in pairs");
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore int_mode =
        (SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per row pair: the TT_SFPLOADs are hoisted ahead of the recorded compute
    // (the int->fp cast doesn't touch the B registers so reordering is safe),
    // replay the body, then two SFPSTOREs under ADDR_MOD_2 which auto-advance
    // dst_reg for the next pair.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        _quant_load_row_pair_<int_mode>(in0_off, in1_off);  // A (int32 -> sign-magn), B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80 (excess-128), both rows
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN);  // CAST + MAD + STOCH_RND, r0/r1 interleaved
        _quant_store_row_pair_<int_mode>(out_off);              // store (sign-magn -> int_mode bits)
    }
}

// Fold +128.0 into the fp32 zero-point in LREG2 so the per-iteration MAD yields v + 128 directly
inline void _int8_bias_zero_point_() { TTI_SFPADDI(INT8_OFFSET_128_IMM16, p_sfpu::LREG2, 0); }

// v = A * B + LREG2 for both rows of the pair (the second MAD fills the
// first one's 2-cycle shadow).
inline void _quant_mad_row_pair_() {
    TTI_SFPMAD(QUANT_ROW0_A, QUANT_ROW0_B, p_sfpu::LREG2, QUANT_ROW0_A, 0 /*mod1*/);
    TTI_SFPMAD(QUANT_ROW1_A, QUANT_ROW1_B, p_sfpu::LREG2, QUANT_ROW1_A, 0 /*mod1*/);
}

// int32 sign-magnitude (loaded that way regardless of input bit
// representation, via SFPLOAD instr_mod0) -> fp32, both rows.
inline void _quant_cast_row_pair_() {
    TTI_SFPCAST(QUANT_ROW0_A, QUANT_ROW0_A, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
    TTI_SFPCAST(QUANT_ROW1_A, QUANT_ROW1_A, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
}

// fp32 -> int for both rows. LCONST_0 (LREG9) is the HW-provided 0.0 used as
// the zero descale. RND_MODE is FP32_TO_UINT8 (round into the full [0, 255]
// range) or FP32_TO_INT8 (clamp to signed int8 [-128, 127]).
template <std::uint32_t RND_MODE>
inline void _quant_round_row_pair_() {
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0 /*imm8*/, p_sfpu::LCONST_0, QUANT_ROW0_A, QUANT_ROW0_A, RND_MODE);
    TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_EVEN, 0 /*imm8*/, p_sfpu::LCONST_0, QUANT_ROW1_A, QUANT_ROW1_A, RND_MODE);
}

// Clamp / round / xor the MAD results of both rows.
// Low 8 bits of each row's A register hold the 2's complement int8 byte.
inline void _int8_pack_fixup_() {
    // Values below -128 (i.e. v + 128 < 0) must saturate to -128. FP32_TO_UINT8 returns the
    // magnitude of a negative input instead of 0, so clamp these lanes to 0.0 first:
    // FP32_TO_UINT8(0.0) = 0, and 0 ^ 0x80 = 0x80, which is the two's-complement encoding of
    // -128 (the minimum int8 value). Each row gets its own SETCC / MOV / ENCC triple: the
    // SETCC runs on enabled lanes only, so the flags must be re-enabled between the rows.
    TTI_SFPSETCC(0, QUANT_ROW0_A, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, QUANT_ROW0_A, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSETCC(0, QUANT_ROW1_A, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, QUANT_ROW1_A, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    _quant_round_row_pair_<sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8>();  // u = round(v + 128) in [0, 255]
    TTI_SFPXOR(0, p_sfpu::LREG4, QUANT_ROW0_A, 0);                   // b = u ^ 0x80
    TTI_SFPXOR(0, p_sfpu::LREG4, QUANT_ROW1_A, 0);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_quant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output:
    // The MAD + offset-128 pack body is recorded once into QUANT_REPLAY_SLOT and replayed,
    static_assert(ITERATIONS % 2 == 0, "quant kernels process dst rows in pairs");
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        _quant_load_row_pair_<InstrModLoadStore::FP32>(in0_off, in1_off);  // A (fp32), B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN_INT8_OUT);        // MAD + offset-128 pack, two rows
        _quant_store_row_pair_<InstrModLoadStore::INT32_2S_COMP>(out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool INT8_INPUT = false>
inline void calculate_requant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output:
    // The CAST + MAD + offset-128 pack body is recorded once into REQUANT_REPLAY_SLOT and replayed.
    // The int8-input unbias (byte ^ 0x80) stays inline before the replay.
    static_assert(ITERATIONS % 2 == 0, "quant kernels process dst rows in pairs");
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        _quant_load_row_pair_<InstrModLoadStore::INT32_2S_COMP>(in0_off, in1_off);  // A (int32/byte), B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80, both rows
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN_INT8_OUT);  // CAST + MAD + offset-128 pack, two rows
        _quant_store_row_pair_<InstrModLoadStore::INT32_2S_COMP>(out_off);
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
    static_assert(ITERATIONS % 2 == 0, "quant kernels process dst rows in pairs");
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr InstrModLoadStore in_mode =
        (SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) ? InstrModLoadStore::INT32 : InstrModLoadStore::INT32_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per row pair: hoist the TT_SFPLOADs ahead of the recorded compute,
    // replay the body, then two SFPSTOREs under ADDR_MOD_2 which auto-advance
    // dst_reg for the next pair.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        _quant_load_row_pair_<in_mode>(in0_off, in1_off);  // A (int32 -> sign-magn), B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80, both rows
        }
        lltt::replay(DEQUANT_REPLAY_SLOT, DEQUANT_REPLAY_LEN);     // CAST + ADD + MUL, r0/r1 interleaved
        _quant_store_row_pair_<InstrModLoadStore::FP32>(out_off);  // store fp32
    }
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT /*unused*/ = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32>
void quant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "quant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
        _int8_bias_zero_point_();  // fold +128 into the fp32 zero-point in LREG2
        _quant_kernels_configure_dest_incr_addrmod_();
        // Record the int8 body (MAD + offset-128 pack), two rows per replay
        lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN_INT8_OUT);
        {
            _quant_mad_row_pair_();  // v = A * B + (zp + 128)
            _int8_pack_fixup_();
        }
        return;
    }
    _quant_kernels_configure_dest_incr_addrmod_();

    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN);
    {
        // D(A) = A * B + LREG2 (zero point), both rows.
        // SFPMAD has a 2-cycle write latency; the second row's MAD fills the
        // slot that used to hold an SFPNOP, so each SFP_STOCH_RND below still
        // issues two slots after the MAD it reads. The recorded body contains
        // only SFPU-pipe opcodes (a hard requirement for replay-buffer
        // playback - the replay buffer feeds the SFPU pipe directly).
        _quant_mad_row_pair_();
        // fp32 -> int. For unsigned (uint8) output, round into the full
        // [0, 255] range; otherwise clamp to signed int8 [-128, 127].
        if constexpr (OUTPUT_FORMAT == DataFormat::UInt8) {
            _quant_round_row_pair_<sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8>();
        } else {
            _quant_round_row_pair_<sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8>();
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
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (INT8_INPUT || OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG4, INT8_SIGN_MASK);
    }
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
        _int8_bias_zero_point_();  // fold +128 into the fp32 zero-point in LREG2
        _quant_kernels_configure_dest_incr_addrmod_();
        // Record the int8 body (CAST + MAD + offset-128 pack), two rows per replay
        lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN_INT8_OUT);
        {
            _quant_cast_row_pair_();  // int32 -> fp32
            _quant_mad_row_pair_();   // v = A * B + (zp + 128)
            _int8_pack_fixup_();
        }
        return;
    }
    _quant_kernels_configure_dest_incr_addrmod_();

    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN);
    {
        // int32 sign-magnitude -> fp32, both rows.
        _quant_cast_row_pair_();
        // D(A) = A * B + LREG2 (zero point), both rows. The second row's MAD
        // fills the first MAD's 2-cycle write-latency slot (formerly an
        // SFPNOP); each SFP_STOCH_RND still reads its A two slots after the MAD.
        _quant_mad_row_pair_();
        // fp32 -> int. For unsigned (uint8) output, round into the full
        // [0, 255] range; otherwise clamp to signed int8 [-128, 127].
        if constexpr (OUTPUT_FORMAT == DataFormat::UInt8) {
            _quant_round_row_pair_<sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8>();
        } else {
            _quant_round_row_pair_<sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8>();
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
        // int32 sign-magnitude -> fp32, both rows.
        _quant_cast_row_pair_();
        // SFPADD = VA*VB + VC ; with LCONST_1 (LREG10) = 1.0 this collapses
        // to A + LREG2 (= A + zero_point as loaded by the caller). The second
        // row's SFPADD fills the first one's 2-cycle write-latency slot.
        TTI_SFPADD(QUANT_ROW0_A, p_sfpu::LCONST_1, p_sfpu::LREG2, QUANT_ROW0_A, 0 /*mod1*/);
        TTI_SFPADD(QUANT_ROW1_A, p_sfpu::LCONST_1, p_sfpu::LREG2, QUANT_ROW1_A, 0 /*mod1*/);
        // SFPMUL with LCONST_0 (LREG9 = 0.0) ignored as +C :
        // A = (A + LREG2) * B, both rows. The first SFPSTORE that follows the
        // replay issues two slots after the row-0 SFPMUL (the row-1 SFPMUL sits
        // between them) and the second store two slots after the row-1 SFPMUL,
        // so no trailing SFPNOP is needed.
        TTI_SFPMUL(QUANT_ROW0_A, QUANT_ROW0_B, p_sfpu::LCONST_0, QUANT_ROW0_A, 0 /*mod1*/);
        TTI_SFPMUL(QUANT_ROW1_A, QUANT_ROW1_B, p_sfpu::LCONST_0, QUANT_ROW1_A, 0 /*mod1*/);
    }
}

}  // namespace ckernel::sfpu
