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
// kernel uses the matching REPLAY_LEN. The OUTPUT_FORMAT == UInt8 (uint8 output)
// quant/requant variant safely shares the slot for the same reason: it only changes the
// clamp constants of _rne_clamp_convert_, not the body length. Distinct slots between kernels
// are required so a single compute kernel can mix all three ops without each init clobbering
// the others' recordings.
//
// Body content (see the inits for the exact emission order). On Blackhole the SFPU implicitly
// stalls on read-after-write hazards between back-to-back fp32 ops, so SFPMAD->SFPADD,
// SFPADD->SFPMUL and SFPMUL->SFPSTORE don't need explicit pipeline bubbles. SFPSWAP is the
// exception: it does not wait for an SFPMAD / SFPADD result, so one instruction must separate them
// (ckernel_sfpu_exp.h fills that slot with independent work; these bodies have none, hence the SFPNOP).
// <clamp + convert> is SFPSWAP (max lo), SFPSWAP (min hi), SFPIADD (see RNE_MAGIC_FP32).
//   QUANT   ( 2s-comp, 5 ) : SFPMAD, SFPNOP, <clamp + convert>                      (int32 / uint8 output)
//   QUANT   (sign-magn, 2) : SFPMAD, STOCH_RND
//   QUANT   (int8-out, 6 ) : SFPMAD, SFPNOP, <clamp + convert>, SFPXOR
//   REQUANT ( 2s-comp, 9 ) : SFPCAST+SFPSETSGN(in), SFPCAST(int->fp32), SFPMAD, SFPADD, SFPNOP,
//                            <clamp + convert>                                     (int32 / uint8 output)
//   REQUANT (sign-magn, 3) : SFPCAST(int->fp32), SFPMAD, STOCH_RND
//   REQUANT (int8-in,  7 ) : SFPCAST(int->fp32), SFPMAD, SFPADD, SFPNOP, <clamp + convert>
//   REQUANT (int8-out, 8 ) : SFPCAST(int->fp32), SFPMAD, SFPADD, SFPNOP, <clamp + convert>, SFPXOR (int8 input)
//   REQUANT (int8-out, 10) : SFPCAST+SFPSETSGN(in), <the 8 above>                                (int32 input)
//   DEQUANT ( 2s-comp, 5 ) : SFPCAST+SFPSETSGN(in), SFPCAST(int->fp32),
//                             SFPADD, SFPMUL
//   DEQUANT (sign-magn, 3) : SFPCAST(int->fp32), SFPADD, SFPMUL
constexpr std::uint32_t QUANT_REPLAY_SLOT = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN_2S_COMP = 5;
constexpr std::uint32_t QUANT_REPLAY_LEN_SIGN_MAGN = 2;
constexpr std::uint32_t QUANT_REPLAY_LEN_INT8_OUT = 6;
constexpr std::uint32_t QUANT_REPLAY_LEN_MAX = QUANT_REPLAY_LEN_INT8_OUT;

constexpr std::uint32_t REQUANT_REPLAY_SLOT = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_MAX;
constexpr std::uint32_t REQUANT_REPLAY_LEN_2S_COMP = 9;
constexpr std::uint32_t REQUANT_REPLAY_LEN_SIGN_MAGN = 3;
constexpr std::uint32_t REQUANT_REPLAY_LEN_INT8_IN = 7;
constexpr std::uint32_t REQUANT_REPLAY_LEN_INT8_OUT = 8;
constexpr std::uint32_t REQUANT_REPLAY_LEN_INT8_OUT_INT32_IN = 10;
constexpr std::uint32_t REQUANT_REPLAY_LEN_MAX = REQUANT_REPLAY_LEN_INT8_OUT_INT32_IN;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_MAX;
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
// get its byte (see the Int8 L1 pack path above). The trailing INT32_2S_COMP SFPSTORE is a no-op on BH, so n
// lands in Dest as is.
constexpr std::uint32_t RNE_MAGIC_FP32 = 0x4b400000u;  // 12582912.0f = 1.5 * 2^23

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
inline void _int8_input_unbias_() { TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0); }

// OUTPUT_FORMAT selects the quantized output tensor dtype: Int32 (the default), UInt8, or Int8. All
// three round to nearest even and saturate through _rne_clamp_convert_: Int32 holds int8-range values
// [-128, 127] in an int32 container, UInt8 [0, 255], and Int8 packs the excess-128 byte.
//
// SIGN_MAGNITUDE_FORMAT = true keeps the previous MAD + FP32_TO_INT8 body (ties away from zero,
// saturates at +/-127). quant_tile never instantiates it.
template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32>
void quant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "quant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    // One-time setup for calculate_quant_int32:
    //   1. load the fp32 zero-point constant into LREG2 (and the RNE constants, see below);
    //   2. program ADDR_MOD_6 with dest+=2 for the per-iteration SFPSTORE;
    //   3. record the register-only compute body into the SFPU replay buffer
    //      under QUANT_REPLAY_SLOT (NoExec - we don't want to issue SFPMAD/
    //      STOCH_RND against undefined LREG0/LREG1 contents at record time).
    // Subsequent _quant_int32_ calls replay the recorded body, shrinking the
    // unrolled binary from ~ITERATIONS*REPLAY_LEN body instructions down to
    // one replay invocation per iteration.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8 || !SIGN_MAGNITUDE_FORMAT) {
        // LREG5 = RNE_MAGIC (the MAD addend); LREG2 = t = RNE_MAGIC - zero-point, exact for an integer
        // zero point, which _rne_clamp_init_ turns into the clamp bounds and then -bits(t) (+128).
        _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_neg1, p_sfpu::LREG5, p_sfpu::LREG2, 0 /*mod1*/);
        _rne_clamp_init_<OUTPUT_FORMAT, p_sfpu::LREG2>();
        if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
            _sfpu_load_imm32_(p_sfpu::LREG3, INT8_SIGN_MASK);
        }
        constexpr std::uint32_t REPLAY_LEN =
            OUTPUT_FORMAT == DataFormat::Int8 ? QUANT_REPLAY_LEN_INT8_OUT : QUANT_REPLAY_LEN_2S_COMP;
        lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, REPLAY_LEN);
        {
            // m = RNE(A * B) + RNE_MAGIC: the single rounding step of the MAD rounds to nearest even
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /*mod1*/);
            TTI_SFPNOP;                            // SFPSWAP does not wait for the SFPMAD result
            _rne_clamp_convert_<p_sfpu::LREG2>();  // n = clamp(RNE(A * B) + zp) (+128)
            if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
                TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // b = (n + 128) ^ 0x80
            }
        }
        return;
    }

    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN_SIGN_MAGN);
    {
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point). The Blackhole SFPU
        // implicitly stalls SFP_STOCH_RND below until SFPMAD's LREG0 write
        // retires, so no explicit pipeline-bubble TTI_SFPNOP is needed here.
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // fp32 -> int. LCONST_0 (LREG9) is the HW-provided 0.0 used as the zero
        // descale. For unsigned (uint8) output, round into the full [0, 255]
        // range, else clamp to sign-magnitude [-127, 127].
        if constexpr (OUTPUT_FORMAT == DataFormat::UInt8) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0 /*imm8*/,
                p_sfpu::LCONST_0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        } else {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0 /*imm8*/,
                p_sfpu::LCONST_0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        }
    }
}

template <
    bool APPROXIMATION_MODE /*unused*/,
    bool SIGN_MAGNITUDE_FORMAT = false,
    DataFormat OUTPUT_FORMAT = DataFormat::Int32,
    bool INT8_INPUT = false>
void requant_init(const uint zero_point) {
    static_assert(
        OUTPUT_FORMAT == DataFormat::Int32 || OUTPUT_FORMAT == DataFormat::UInt8 || OUTPUT_FORMAT == DataFormat::Int8,
        "requant_init OUTPUT_FORMAT must be Int32, UInt8 or Int8");
    // One-time setup for requant_int32; see quant_init for the
    // record/replay rationale. Loads the zero point into LREG2 (and the RNE
    // constants, see below), programs ADDR_MOD_6 with dest+=2, then records
    // the register-only compute into REQUANT_REPLAY_SLOT (NoExec).
    //
    // The body rounds the whole expression q * (s_in / s_out) + zp to nearest even,
    // zp being the host-folded z_out - z_in * (s_in / s_out); this matches the op's
    // golden round((q - z_in) * (s_in / s_out) + z_out).
    //
    // SIGN_MAGNITUDE_FORMAT = true keeps the previous CAST + MAD + FP32_TO_INT8 body (int32
    // input and output only). requant_tile never instantiates it.
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (INT8_INPUT || OUTPUT_FORMAT == DataFormat::Int8) {
        _sfpu_load_imm32_(p_sfpu::LREG3, INT8_SIGN_MASK);
    }
    _quant_kernels_configure_dest_incr_addrmod_();
    if constexpr (OUTPUT_FORMAT == DataFormat::Int8 || !SIGN_MAGNITUDE_FORMAT) {
        // LREG5 = RNE_MAGIC (the ADD addend); with t = RNE_MAGIC, LREG6 ends up as -bits(t) (+128).
        _sfpu_load_imm32_(p_sfpu::LREG5, RNE_MAGIC_FP32);
        _sfpu_load_imm32_(p_sfpu::LREG6, RNE_MAGIC_FP32);
        _rne_clamp_init_<OUTPUT_FORMAT, p_sfpu::LREG6>();
        // Int8 input is unbiased (byte ^ 0x80) inline by the kernel before the replay. Int32
        // input runs the 2's-complement -> sign-magnitude fixup inside the recorded body.
        constexpr std::uint32_t REPLAY_LEN =
            OUTPUT_FORMAT == DataFormat::Int8
                ? (INT8_INPUT ? REQUANT_REPLAY_LEN_INT8_OUT : REQUANT_REPLAY_LEN_INT8_OUT_INT32_IN)
                : (INT8_INPUT ? REQUANT_REPLAY_LEN_INT8_IN : REQUANT_REPLAY_LEN_2S_COMP);
        lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REPLAY_LEN);
        {
            if constexpr (!INT8_INPUT) {
                // Input arrives in 2's-complement bits in LREG0 (the upstream
                // quant kernel stores 2's-complement, and the INT32_2S_COMP SFPLOAD
                // mode is a no-op on BH). Convert to sign-magnitude so the int->fp32
                // SFPCAST below sees its expected input. See INT_REPR_SWAP_CAST above.
                //
                // Skipped for INT8_INPUT: the byte comes through the UInt8 unpacker
                // and is unbiased to excess-128 (non-negative) inline by the kernel,
                // so no sign-magnitude fixup is needed before the SFPCAST.
                apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG4, INT_REPR_SWAP_CAST);
            }
            TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);  // int -> fp32
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);  // v = A * B + zp
            TTI_SFPADD(
                p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG0, 0 /*mod1*/);  // m = RNE(v) + MAGIC
            TTI_SFPNOP;                            // SFPSWAP does not wait for the SFPADD result
            _rne_clamp_convert_<p_sfpu::LREG6>();  // n = clamp(RNE(v)) (+128)
            if constexpr (OUTPUT_FORMAT == DataFormat::Int8) {
                TTI_SFPXOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // b = (n + 128) ^ 0x80
            }
        }
        return;
    }

    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REQUANT_REPLAY_LEN_SIGN_MAGN);
    {
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point). BH SFPU implicitly
        // stalls STOCH_RND below until SFPMAD's LREG0 write retires, so no
        // explicit pipeline-bubble TTI_SFPNOP is needed here.
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0 /*mod1*/);
        // fp32 -> int. LCONST_0 (LREG9) provides the 0.0 descale. For unsigned
        // (uint8) output, round into the full [0, 255] range; otherwise clamp to
        // sign-magnitude [-127, 127].
        if constexpr (OUTPUT_FORMAT == DataFormat::UInt8) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0 /*imm8*/,
                p_sfpu::LCONST_0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        } else {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0 /*imm8*/,
                p_sfpu::LCONST_0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
void dequant_init(const uint zero_point) {
    // One-time setup for calculate_dequant_int32; see quant_init for the
    // record/replay rationale. The caller passes -zero_point (so the
    // recorded body computes (A + LREG2) * B = (A - zero_point) * B).
    _sfpu_load_imm32_(p_sfpu::LREG2, zero_point);
    if constexpr (INT8_INPUT) {
        _sfpu_load_imm32_(p_sfpu::LREG3, INT8_SIGN_MASK);
    }
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN =
        (SIGN_MAGNITUDE_FORMAT || INT8_INPUT) ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(DEQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        if constexpr (!SIGN_MAGNITUDE_FORMAT && !INT8_INPUT) {
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
        lltt::replay(QUANT_REPLAY_SLOT, REPLAY_LEN);                              // RNE MAD + pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);  // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void calculate_requant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A is input to requant (int32, sign-magnitude or 2's complement bits or UInt8-unpacked int8 byte in [0,
    // 255]). Operand B is scaling factor (fp32). LREG2 holds the zero-point constant (fp32) loaded by
    // _init_requant_int32_. Output is int32 scaled to int8 range (sign-magnitude or 2's-complement).
    //
    // The replay-buffer body at REQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_requant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT, OUTPUT_FORMAT, INT8_INPUT>, which must run before
    // the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN =
        INT8_INPUT ? REQUANT_REPLAY_LEN_INT8_IN
                   : (SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP);

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute
    // (the input cast doesn't touch LREG1 so reordering is safe), replay the
    // recorded body, then SFPSTORE under ADDR_MOD_6 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off);  // operand A (int32/byte)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);           // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REPLAY_LEN);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);  // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_quant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output: RNE MAD + offset-128 pack body is recorded once into QUANT_REPLAY_SLOT and replayed.
    constexpr std::uint32_t dst_tile_size = 64;
    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, in0_off);  // operand A (fp32)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);  // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, QUANT_REPLAY_LEN_INT8_OUT);               // RNE MAD + offset-128 pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool INT8_INPUT = false>
inline void calculate_requant_int32_int8_pack(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Int8 output: CAST + RNE MAD + offset-128 pack body is recorded once into REQUANT_REPLAY_SLOT and replayed.
    constexpr std::uint32_t dst_tile_size = 64;
    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    constexpr std::uint32_t REPLAY_LEN =
        INT8_INPUT ? REQUANT_REPLAY_LEN_INT8_OUT : REQUANT_REPLAY_LEN_INT8_OUT_INT32_IN;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off);  // operand A (int32/byte)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);           // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80 (int32 input is converted inside the recorded body)
        }
        lltt::replay(REQUANT_REPLAY_SLOT, REPLAY_LEN);  // [int32 input: 2's-comp -> sign-mag] + CAST + RNE MAD + pack
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false, bool INT8_INPUT = false>
inline void calculate_dequant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Operand A[LREG0] is input to dequant (int32, sign-magnitude or 2's complement bits;
    // or, when INT8_INPUT, a UInt8-unpacked int8 byte in [0, 255]).
    // Operand B[LREG1] is scaling factor (fp32).
    // LREG2 holds the (negated) zero-point constant loaded by _init_dequant_int32_;
    // i.e. the formula computed is (A + LREG2) * B, which is (A - zero_point) * B
    // when the caller passes -zero_point through the init.
    //
    // The replay-buffer body at DEQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_dequant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT, INT8_INPUT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    // INT8_INPUT reuses the sign-magnitude body (no input int-repr conversion).
    constexpr std::uint32_t REPLAY_LEN =
        (SIGN_MAGNITUDE_FORMAT || INT8_INPUT) ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute,
    // replay the body, then SFPSTORE under ADDR_MOD_6 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off);  // operand A (int32/byte)
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, in1_off);           // operand B (fp32 scaler)
        if constexpr (INT8_INPUT) {
            _int8_input_unbias_();  // byte ^ 0x80
        }
        lltt::replay(DEQUANT_REPLAY_SLOT, REPLAY_LEN);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_6, out_off);  // store fp32 + dst_reg += 2
    }
}

}  // namespace ckernel::sfpu
