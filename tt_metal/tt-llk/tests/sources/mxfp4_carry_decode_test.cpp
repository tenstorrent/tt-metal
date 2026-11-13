// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MXFP4 stored as relabelled bfloat4_b, restored by the SFPU carry decode.
//
// Unpack A (Bfp4_b) -> datacopy to Dest -> SFPU on Pack: bits += 3 * (bits & 0x00200000) on the FP32
// LReg image -> pack. A value with the mantissa's 0.25 bit set (1.25 * 2^e, 1.75 * 2^e) drops that bit
// and gains one exponent step (2^(e+1), 1.5 * 2^(e+1)), which maps the stored codes 5 and 7 (in units of
// half the MXFP4 scale) back to the e2m1 values 8 and 12.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

// Math only stages the tile into Dest; the decode runs on Pack, as in the Blaze kernel.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"
#include "sfpi.h"

namespace
{
// sfpi decoder: the same body as Blaze's mxfp4_remap, over a whole tile (32 vectors of 32 lanes).
// It compiles to load, and, iadd, shft, iadd, store per vector, replayed with the row step.
inline void mxfp4_carry_decode_tile()
{
    const sfpi::vInt mask = 0x00200000;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++)
    {
        sfpi::vInt b = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        sfpi::vInt m = b & mask;
        b            = b + m + (m << 1);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(b);
        sfpi::dst_reg++;
    }
}

// SFPLOADMACRO decoder: per vector v, software issues three instructions and the macro schedules two:
//
//   slot 3v    SFPLOADMACRO R = Dst[v]   (R alternates L0 / L4; macro v & 1)
//   slot 3v+1  SFPAND   T = R & 0x00200000                       simple sub-unit
//   slot 3v+2  SFPMUL24 T3[v & 1] = T * 3                         MAD sub-unit (low 23 bits)
//   slot 3v+5  (scheduled) SFPIADD L16 = R + T3[v & 1]           simple sub-unit, delay 4
//   slot 3v+6  (scheduled) SFPSTORE Dst[v] = L16                 store sub-unit, delay 5
//
// The scheduled IADD lands on a MUL24 slot and the scheduled store on a load slot, so neither displaces
// a software instruction (the simple sub-unit runs the software AND on slots 3v+1). Delays count issued
// SFPU instructions. R is read for the last time at 3v+5, before the macro two vectors later reloads it
// at 3v+6; T3[v & 1] is written at 3v+2 and read at 3v+5, three issues after the MAD result (two cycles).
constexpr std::uint32_t kMaskReg  = 3;
constexpr std::uint32_t kThreeReg = 7;
constexpr std::uint32_t kTReg     = 1;
constexpr std::uint32_t kT3Reg0   = 2;
constexpr std::uint32_t kT3Reg1   = 6;
constexpr std::uint32_t kZeroReg  = 9; // constant 0: keeps SFPMUL24's VC exponent zero

inline void mxfp4_macro_configure()
{
    constexpr std::uint32_t iadd_reg_reg = 4;   // SFPIADD: VD = VC + VD, condition codes untouched
    TTI_SFPCONFIG(0, 0xF, 1 /*imm*/);           // LaneConfig defaults: backdoor template loads enabled
    TTI_SFPNOP;
    // An instruction with VD = 12 + i is stored as InstructionTemplate[i] instead of executing.
    TTI_SFPIADD(0, kT3Reg0, 12, iadd_reg_reg);
    TTI_SFPNOP;
    TTI_SFPIADD(0, kT3Reg1, 13, iadd_reg_reg);
    TTI_SFPNOP;
    // Sequence bytes: simple unit in bits 0..7, store unit in bits 24..31. Bit 7 = take VB from the
    // macro's register, bit 6 = write (simple) / read (store) LReg16, bits 5..3 = delay, bits 2..0 =
    // 4 + template index (simple) or 3 = SFPSTORE (store).
    constexpr std::uint32_t store_byte = 0x40 | (5 << 3) | 3;
    constexpr std::uint32_t seq0       = (store_byte << 24) | (0x80 | 0x40 | (4 << 3) | 4);
    constexpr std::uint32_t seq1       = (store_byte << 24) | (0x80 | 0x40 | (4 << 3) | 5);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq0 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq0 >> 16);
    TTI_SFPCONFIG(0, 4 + 0, 0 /*from LReg0*/);
    TTI_SFPNOP;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq1 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq1 >> 16);
    TTI_SFPCONFIG(0, 4 + 1, 0 /*from LReg0*/);
    TTI_SFPNOP;
    // Misc: StoreMod0 = 0; macros 0 and 1 store with their load's Mod0 (bits 4, 5); the simple and
    // store sub-units count delays in issued instructions (bits 8, 11).
    TTI_SFPCONFIG(0x930, 8, 1 /*imm*/);
    TTI_SFPNOP;
}

inline void mxfp4_macro_decode_tile()
{
    constexpr std::uint32_t fmt_implied    = 0;
    constexpr std::uint32_t addr_no_incr   = 7;
    constexpr std::uint32_t and_use_vb     = 1;
    constexpr std::uint32_t mul24_lower    = 0;
    constexpr std::uint32_t chain[2]       = {0, 4};
    constexpr std::uint32_t t3[2]          = {kT3Reg0, kT3Reg1};
    TTI_SFPLOADI(kMaskReg, 0 /*BF16 immediate: upper half*/, 0x0020); // 0x00200000
    TTI_SFPLOADI(kThreeReg, sfpi::SFPLOADI_MOD0_LOWER, 3);
    TTI_SFPLOADI(kThreeReg, sfpi::SFPLOADI_MOD0_UPPER, 0);
#pragma GCC unroll 32
    for (std::uint32_t v = 0; v < 32; ++v)
    {
        const std::uint32_t r = chain[v & 1];
        // The address's low bit is unused by the load; it carries bit 2 of the register index.
        TTI_SFPLOADMACRO(((v & 1) << 2) | (r & 3), fmt_implied, addr_no_incr, v * 2 + (r >> 2));
        TTI_SFPAND(r /*VB*/, kMaskReg, kTReg, and_use_vb);
        TTI_SFPMUL24(kTReg, kThreeReg, kZeroReg, t3[v & 1], mul24_lower);
    }
    // Let the last two vectors' scheduled IADD and store count down and run.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// Two-load SFPLOADMACRO decoder: per vector v, software issues two SFPLOADMACROs of the same Dst row and
// the macros schedule everything else:
//
//   slot 2v    SFPLOADMACRO A[v & 1] = Dst[v]   (macro 0)
//   slot 2v+1  (scheduled) SFPAND   A = A & 0x00200000               simple sub-unit, delay 0
//   slot 2v+1  SFPLOADMACRO B[v & 1] = Dst[v]   (macro 1 + (v & 1))
//   slot 2v+2  (scheduled) SFPMUL24 A = A * 3                        MAD sub-unit, delay 1
//   slot 2v+4  (scheduled) SFPIADD  L16 = B + A[v & 1]               simple sub-unit, delay 2
//   slot 2v+5  (scheduled) SFPSTORE Dst[v] = L16                     store sub-unit, delay 3
//
// The ANDs land on odd slots and the IADDs on even ones, so the simple sub-unit takes one of each per two
// slots. The IADD names A[v & 1] through its template (one template per A register), so A rotates over
// two registers only, and the macro of vector v + 2 reloads A[v & 1] on slot 2v+4, the very slot the IADD
// of vector v reads it: this relies on a scheduled instruction reading its LRegs before an SFPLOADMACRO
// issued on the same cycle writes them (not documented; the bit-exact test checks it).
constexpr std::uint32_t kA[2] = {0, 4};
constexpr std::uint32_t kB[2] = {1, 5};

inline void mxfp4_macro2_configure()
{
    constexpr std::uint32_t iadd_reg_reg = 4; // SFPIADD: VD = VC + VD, condition codes untouched
    TTI_SFPCONFIG(0, 0xF, 1 /*imm*/);         // LaneConfig defaults: backdoor template loads enabled
    TTI_SFPNOP;
    // InstructionTemplate[0..3] via VD = 12 + i.
    TTI_SFPAND(0 /*VB, from the macro*/, kMaskReg, 12, 1 /*USE_VB*/);
    TTI_SFPNOP;
    TTI_SFPMUL24(kThreeReg, 0 /*VB, from the macro*/, kZeroReg, 13, 0 /*lower*/);
    TTI_SFPNOP;
    TTI_SFPIADD(0, kA[0], 14, iadd_reg_reg);
    TTI_SFPNOP;
    TTI_SFPIADD(0, kA[1], 15, iadd_reg_reg);
    TTI_SFPNOP;
    // Sequence bytes: simple unit bits 0..7, MAD 8..15, store 24..31. Bit 7 = VB from the macro's register,
    // bit 6 = write (simple) / read (store) LReg16, bits 5..3 = delay, bits 2..0 = 4 + template or 3 = store.
    constexpr std::uint32_t seq_a  = ((0x80 | (1 << 3) | 5) << 8) | (0x80 | (0 << 3) | 4);
    constexpr std::uint32_t store  = 0x40 | (3 << 3) | 3;
    constexpr std::uint32_t seq_b0 = (store << 24) | (0x80 | 0x40 | (2 << 3) | 6);
    constexpr std::uint32_t seq_b1 = (store << 24) | (0x80 | 0x40 | (2 << 3) | 7);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq_a & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq_a >> 16);
    TTI_SFPCONFIG(0, 4 + 0, 0 /*from LReg0*/);
    TTI_SFPNOP;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq_b0 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq_b0 >> 16);
    TTI_SFPCONFIG(0, 4 + 1, 0 /*from LReg0*/);
    TTI_SFPNOP;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, seq_b1 & 0xFFFF);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, seq_b1 >> 16);
    TTI_SFPCONFIG(0, 4 + 2, 0 /*from LReg0*/);
    TTI_SFPNOP;
    // Misc: StoreMod0 = 0; macros 1 and 2 store with their load's Mod0 (bits 5, 6); every sub-unit counts
    // delays in issued instructions (bits 8..11).
    TTI_SFPCONFIG(0xF60, 8, 1 /*imm*/);
    TTI_SFPNOP;
}

inline void mxfp4_macro2_decode_tile()
{
    constexpr std::uint32_t fmt_implied  = 0;
    constexpr std::uint32_t addr_no_incr = 7;
    TTI_SFPLOADI(kMaskReg, 0 /*BF16 immediate: upper half*/, 0x0020); // 0x00200000
    TTI_SFPLOADI(kThreeReg, sfpi::SFPLOADI_MOD0_LOWER, 3);
    TTI_SFPLOADI(kThreeReg, sfpi::SFPLOADI_MOD0_UPPER, 0);
#pragma GCC unroll 32
    for (std::uint32_t v = 0; v < 32; ++v)
    {
        const std::uint32_t a = kA[v & 1];
        const std::uint32_t b = kB[v & 1];
        // The address's low bit is unused by the load; it carries bit 2 of the register index.
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), fmt_implied, addr_no_incr, v * 2 + (a >> 2));
        TTI_SFPLOADMACRO(((1 + (v & 1)) << 2) | (b & 3), fmt_implied, addr_no_incr, v * 2 + (b >> 2));
    }
    // Let the last vectors' scheduled IADD and store count down and run.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
#if defined(MXFP4_DECODER_MACRO2)
    mxfp4_macro2_configure();
#elif defined(MXFP4_DECODER_MACRO)
    mxfp4_macro_configure();
#endif

    // Timing: wall-clock cycles around each tile's decode, both ends after the thread has drained
    // (tensix_sync), and the same bracket around nothing to calibrate the bracket itself.
    std::uint64_t decode_cycles = 0;
    std::uint64_t empty_cycles  = 0;
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();

        _llk_math_eltwise_sfpu_start_(0 /* dst_index */);
        ckernel::tensix_sync();
        const std::uint64_t e0 = ckernel::read_wall_clock();
        ckernel::tensix_sync();
        const std::uint64_t e1 = ckernel::read_wall_clock();
        ckernel::tensix_sync();
        const std::uint64_t t0 = ckernel::read_wall_clock();
#if defined(MXFP4_DECODER_MACRO2)
        mxfp4_macro2_decode_tile();
#elif defined(MXFP4_DECODER_MACRO)
        mxfp4_macro_decode_tile();
#else
        mxfp4_carry_decode_tile();
#endif
        ckernel::tensix_sync();
        const std::uint64_t t1 = ckernel::read_wall_clock();
        decode_cycles += t1 - t0;
        empty_cycles += e1 - e0;
        _llk_math_eltwise_sfpu_done_();

        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
    // The tile after the last result tile holds the timing words (read back as raw BF16 pairs).
    ckernel::tensix_sync();
    volatile std::uint32_t* timing = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[params.TILE_CNT]);
    timing[0]                      = static_cast<std::uint32_t>(decode_cycles);
    timing[1]                      = static_cast<std::uint32_t>(empty_cycles);
    timing[2]                      = params.TILE_CNT;
    timing[3]                      = 0x4D584650; // "MXFP" marker
}

#endif
