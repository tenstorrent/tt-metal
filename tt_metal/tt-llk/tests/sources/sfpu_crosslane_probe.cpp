// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Cross-lane arsenal: permutation-identity LANE TRACER probe (lane FB).
//
// Structure mirrors sfpu_lreg_ladder_probe.cpp (lane DS's proven custom-body
// + multi-tile-pack precedent): UInt32 unpack-to-dest input, raw-builtin +
// bridge-style bodies, PROBE_MODE if-constexpr dispatch.  Each mode drives
// ONE cross-lane mechanism with data loaded from the input tile and stores
// the results to output tile 3; the python side (test_crosslane_lane_tracer
// .py) calibrates the tensor<->(row,lane) mapping empirically and compares
// EVERY lane against the host oracle (helpers/crosslane_oracle.py), which is
// transcribed from the tt-isa-documentation functional models.
//
// PROBE_MODE map (keep in sync with the python module):
//   0 identity     : out rows 0..15 = in rows 0..15          (calibration)
//   1 rowtag       : out row i = 0x00A00000 + i              (calibration)
//   2 lanetag      : out rows 0..15 = vConstTileId           (calibration)
//   3 transp8      : SFPTRANSP over 8 regs (both banks)
//   4 rot family   : ror1 / shr1 / rotr^3 / ror1^8
//   5 copy4        : SFPSHFT2 Mod1=0 queue shuffle
//   6 chained_copy4: SFPSHFT2 Mod1=1 (crosses subvector rows)
//   7 ror1_and_copy4: SFPSHFT2 Mod1=2
//   8 swap mods 1,2,3,4 on four pairs
//   9 swap mods 5,6,7,8 on four pairs
//  10 swap mod 0 (unconditional) + repeat mod 1
//  11 EXCHANGE_SRCB_SRCC global direction flip around a mod-1 swap
//  12 SFPCONFIG lane-mask form (Mod1=8) per-column direction flip
//  13 indexed swap under ENABLE_DEST_INDEX (keys L0/L1, companions L4/L5)
//  14 SFPCONFIG vertical broadcast into LReg[11] / LReg[12]
//  15 integer reduce composition: row fold tree + cross-row via transp +
//     rowvec broadcast via SFPCONFIG dest 11
//  16 register-axis sort4 network (swap mod 1 x5)
//  17 transp-sandwiched sort4 (transp8 . sort4 . transp8)
//  18 partial LO16/HI16 companion roundtrip (+ raw 32b view of the halves)
//  30 BF16 store onto known INT32 content (RMW probe, tt-blaze #2475)

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
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
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"
using namespace ckernel;

// The sfpi headers wrap several builtins in arity-reducing macros that
// splice in ckernel::instrn_buffer; the probe bodies use the RAW builtin
// arity with a nullptr iptr (compile-time token; codegen emits direct
// Tensix mnemonics -- lane DS's proven pattern).  Drop the macros here.
#ifdef __builtin_rvtt_sfpload
#undef __builtin_rvtt_sfpload
#endif
#ifdef __builtin_rvtt_sfpload_lv
#undef __builtin_rvtt_sfpload_lv
#endif
#ifdef __builtin_rvtt_sfpstore
#undef __builtin_rvtt_sfpstore
#endif
#ifdef __builtin_rvtt_sfpxloadi
#undef __builtin_rvtt_sfpxloadi
#endif

namespace
{

// INT32 raw Dst access, NOINC addressing.  Input rows at addr 2*i,
// output rows at 192 + 2*i (tile 3) -- lane DS's proven constants.
constexpr unsigned FMT_I32 = 4;   // MOD0_FMT_INT32
constexpr unsigned FMT_BF16 = 2;  // MOD0_FMT_BF16
constexpr unsigned FMT_U16 = 6;   // MOD0_FMT_UINT16
constexpr unsigned FMT_LO16_ONLY = 14;
constexpr unsigned FMT_HI16_ONLY = 15;
constexpr unsigned NOINC = 7;

#define LDROW(i)     __builtin_rvtt_sfpload(nullptr, 2 * (i), 0, 0, FMT_I32, NOINC)
#define STROW(v, i)  __builtin_rvtt_sfpstore(nullptr, (v), 192 + 2 * (i), 0, 0, FMT_I32, NOINC)

inline void body_identity()
{
#define IDENT(i) STROW(LDROW(i), i)
    IDENT(0); IDENT(1); IDENT(2); IDENT(3);
    IDENT(4); IDENT(5); IDENT(6); IDENT(7);
    IDENT(8); IDENT(9); IDENT(10); IDENT(11);
    IDENT(12); IDENT(13); IDENT(14); IDENT(15);
#undef IDENT
}

inline void body_rowtag()
{
#define ROWTAG(i) STROW(__builtin_rvtt_sfpxloadi(nullptr, 0x00A00000 + (i), 0, 0, 31), i)
    ROWTAG(0); ROWTAG(1); ROWTAG(2); ROWTAG(3);
    ROWTAG(4); ROWTAG(5); ROWTAG(6); ROWTAG(7);
    ROWTAG(8); ROWTAG(9); ROWTAG(10); ROWTAG(11);
    ROWTAG(12); ROWTAG(13); ROWTAG(14); ROWTAG(15);
#undef ROWTAG
}

inline void body_lanetag()
{
    auto v = __builtin_rvtt_sfpreadlreg(15); /* vConstTileId */
#define LANETAG(i) STROW(v, i)
    LANETAG(0); LANETAG(1); LANETAG(2); LANETAG(3);
    LANETAG(4); LANETAG(5); LANETAG(6); LANETAG(7);
    LANETAG(8); LANETAG(9); LANETAG(10); LANETAG(11);
    LANETAG(12); LANETAG(13); LANETAG(14); LANETAG(15);
#undef LANETAG
}

// --- mode 3: SFPTRANSP over all 8 LRegs (both banks) -----------------------
inline void body_transp8()
{
    auto v0 = LDROW(0);
    auto v1 = LDROW(1);
    auto v2 = LDROW(2);
    auto v3 = LDROW(3);
    auto c0 = LDROW(4);
    auto c1 = LDROW(5);
    auto c2 = LDROW(6);
    auto c3 = LDROW(7);
    auto r = __builtin_rvtt_sfptransp8(v0, v1, v2, v3, c0, c1, c2, c3);
    STROW(__builtin_rvtt_sfpselect4(r, 0), 0);
    STROW(__builtin_rvtt_sfpselect4(r, 1), 1);
    STROW(__builtin_rvtt_sfpselect4(r, 2), 2);
    STROW(__builtin_rvtt_sfpselect4(r, 3), 3);
    // companion-bank results ride the fixed-LReg window (the rvtt.md
    // pattern the bridge kit documents for rvtt_sfptransp8)
    STROW(sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg4]).get(), 4);
    STROW(sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg5]).get(), 5);
    STROW(sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg6]).get(), 6);
    STROW(sfpi::vUInt(sfpi::l_reg[sfpi::LRegs::LReg7]).get(), 7);
}

// --- mode 4: rotate family --------------------------------------------------
inline void body_rot()
{
    auto a = LDROW(0);
    auto b = LDROW(1);
    STROW(__builtin_rvtt_sfpshft2_subvec_shfl1(a, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1), 0);
    STROW(__builtin_rvtt_sfpshft2_subvec_shfl1(b, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLSHR1), 1);
    auto r3 = a;
    r3 = __builtin_rvtt_sfpshft2_subvec_shfl1(r3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    r3 = __builtin_rvtt_sfpshft2_subvec_shfl1(r3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    r3 = __builtin_rvtt_sfpshft2_subvec_shfl1(r3, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    STROW(r3, 2);
    auto r8 = b;
    for (int k = 0; k < 8; ++k)
        r8 = __builtin_rvtt_sfpshft2_subvec_shfl1(r8, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    STROW(r8, 3);
}

// --- modes 5/6/7: COPY4 family ---------------------------------------------
// builtin arg order (rvtt.md constraint chain): (l1, l2, l3, l0_or_vc, mod)
// results select4 0..3 = new (L0, L1, L2, L3).
template <unsigned Mod>
inline void body_copy4_family()
{
    if constexpr (Mod == sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4)
    {
        auto r1 = LDROW(1);
        auto r2 = LDROW(2);
        auto r3 = LDROW(3);
        auto vc = LDROW(4);
        auto r = __builtin_rvtt_sfpshft2_subvec_shfl1_copy4(r1, r2, r3, vc, Mod);
        STROW(__builtin_rvtt_sfpselect4(r, 0), 0);
        STROW(__builtin_rvtt_sfpselect4(r, 1), 1);
        STROW(__builtin_rvtt_sfpselect4(r, 2), 2);
        STROW(__builtin_rvtt_sfpselect4(r, 3), 3);
    }
    else if constexpr (Mod == sfpi::SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4)
    {
        // FINDING (lane FB, 2026-08-21): the compiler's spelling of this
        // mode -- __builtin_rvtt_sfpshft2_subvec_copy4 (rvtt.md
        // rvtt_sfpshft2_subvec_copy4_int) -- emits malformed asm
        // "SFPSHFT2 L0 L0, 0, 1" (missing comma) which the BH assembler
        // REJECTS ("extension xtttensixqsr required"): the builtin is
        // unusable on BH at the pinned toolchain.  Probe the INSTRUCTION
        // via an all-raw TTI region instead (no compiled vector values are
        // live, so the physical L0..L3 use cannot clash with the
        // allocator).
        TTI_SFPLOAD(0, FMT_I32, NOINC, 0);
        TTI_SFPLOAD(1, FMT_I32, NOINC, 2);
        TTI_SFPLOAD(2, FMT_I32, NOINC, 4);
        TTI_SFPLOAD(3, FMT_I32, NOINC, 6);
        TTI_SFPSHFT2(0, 0, 0, 1); // SUBVEC_CHAINED_COPY4
        TTI_SFPSTORE(0, FMT_I32, NOINC, 192 + 0);
        TTI_SFPSTORE(1, FMT_I32, NOINC, 192 + 2);
        TTI_SFPSTORE(2, FMT_I32, NOINC, 192 + 4);
        TTI_SFPSTORE(3, FMT_I32, NOINC, 192 + 6);
    }
    else // plain COPY4 (mod 0): 3-source builtin, L3 <- 0
    {
        auto r1 = LDROW(1);
        auto r2 = LDROW(2);
        auto r3 = LDROW(3);
        auto r = __builtin_rvtt_sfpshft2_copy4(r1, r2, r3, Mod);
        STROW(__builtin_rvtt_sfpselect4(r, 0), 0);
        STROW(__builtin_rvtt_sfpselect4(r, 1), 1);
        STROW(__builtin_rvtt_sfpselect4(r, 2), 2);
        STROW(__builtin_rvtt_sfpselect4(r, 3), 3);
    }
}

// --- swap helper -------------------------------------------------------------
template <unsigned Mod>
inline void swap_pair_store(unsigned in_a, unsigned in_b,
                            unsigned out_a, unsigned out_b)
{
    auto a = LDROW(in_a);
    auto b = LDROW(in_b);
    auto r = __builtin_rvtt_sfpswap(a, b, Mod);
    STROW(__builtin_rvtt_sfpselect2(r, 0), out_a);
    STROW(__builtin_rvtt_sfpselect2(r, 1), out_b);
}

inline void body_swap_1234()
{
    swap_pair_store<1>(0, 1, 0, 1);
    swap_pair_store<2>(2, 3, 2, 3);
    swap_pair_store<3>(4, 5, 4, 5);
    swap_pair_store<4>(6, 7, 6, 7);
}

inline void body_swap_5678()
{
    swap_pair_store<5>(0, 1, 0, 1);
    swap_pair_store<6>(2, 3, 2, 3);
    swap_pair_store<7>(4, 5, 4, 5);
    swap_pair_store<8>(6, 7, 6, 7);
}

inline void body_swap_0_and_1()
{
    swap_pair_store<0>(0, 1, 0, 1); // unconditional swap
    swap_pair_store<1>(2, 3, 2, 3);
}

// --- mode 11: global EXCHANGE_SRCB_SRCC flip --------------------------------
inline void body_exchange_flip()
{
    // LaneConfig := 0x0100 (bit 8 = EXCHANGE_SRCB_SRCC), imm value form.
    TTI_SFPCONFIG(0x0100, 0xF, 1);
    swap_pair_store<1>(0, 1, 0, 1);
    // restore LaneConfig := 0
    TTI_SFPCONFIG(0x0000, 0xF, 1);
    // control pair after restore
    swap_pair_store<1>(2, 3, 2, 3);
}

// --- mode 12: SFPCONFIG lane-mask form (Mod1=8) per-column flip -------------
inline void body_lane_masked_flip()
{
    // The hand kernels' phase-7 mechanism (topk_xl): SFPLOADI puts the
    // LaneConfig value in physical L0, then SFPCONFIG Mod1=8 takes the
    // value from LReg[0] and the participating COLUMNS from Imm16 bit
    // (lane&7)*2.  Imm16 0x4444 -> columns 1,3,5,7.
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_USHORT, 0x0100);
    TTI_SFPCONFIG(0x4444, 0xF, 8);
    swap_pair_store<1>(0, 1, 0, 1);
    // restore all columns
    TTI_SFPCONFIG(0x0000, 0xF, 1);
    swap_pair_store<1>(2, 3, 2, 3);
}

// --- mode 13: indexed swap under ENABLE_DEST_INDEX ---------------------------
inline void body_indexed_swap()
{
    auto k0 = LDROW(0);
    auto k1 = LDROW(1);
    auto c0 = LDROW(2);
    auto c1 = LDROW(3);
    // open the window (LaneConfig bit 2); TEN-2932: only SFPLOAD/SFPLOADI/
    // SFPSWAP/SFPTRANSP may write LReg[4..7] while open -- keep it tight.
    TTI_SFPCONFIG(0x0004, 0xF, 1);
    auto r = __builtin_rvtt_sfpswap_indexed(k0, k1, c0, c1, 1);
    // close the window before any compiled moves can touch L4..L7
    TTI_SFPCONFIG(0x0000, 0xF, 1);
    STROW(__builtin_rvtt_sfpselect4(r, 0), 0);
    STROW(__builtin_rvtt_sfpselect4(r, 1), 1);
    STROW(__builtin_rvtt_sfpselect4(r, 2), 2);
    STROW(__builtin_rvtt_sfpselect4(r, 3), 3);
}

// --- mode 14: SFPCONFIG vertical broadcast into LReg[11]/LReg[12] ------------
inline void body_config_broadcast()
{
    auto v0 = LDROW(0);
    __builtin_rvtt_sfpwriteconfig_v(v0, 11);
    STROW(__builtin_rvtt_sfpreadlreg(11), 0);
    auto v1 = LDROW(1);
    __builtin_rvtt_sfpwriteconfig_v(v1, 12);
    STROW(__builtin_rvtt_sfpreadlreg(12), 1);
}

// --- mode 15: integer reduce composition --------------------------------------
inline void body_reduce_int()
{
    // (a) per-row all-lanes fold tree: v = v + ror(v,1); + ror(v,2); + ror(v,4)
    auto v = LDROW(0);
    auto acc = v;
    auto rot = __builtin_rvtt_sfpshft2_subvec_shfl1(acc, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    acc = __builtin_rvtt_sfpiadd_v(rot, acc, sfpi::SFPIADD_MOD1_CC_NONE);
    rot = acc;
    rot = __builtin_rvtt_sfpshft2_subvec_shfl1(rot, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    rot = __builtin_rvtt_sfpshft2_subvec_shfl1(rot, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    acc = __builtin_rvtt_sfpiadd_v(rot, acc, sfpi::SFPIADD_MOD1_CC_NONE);
    rot = acc;
    rot = __builtin_rvtt_sfpshft2_subvec_shfl1(rot, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    rot = __builtin_rvtt_sfpshft2_subvec_shfl1(rot, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    rot = __builtin_rvtt_sfpshft2_subvec_shfl1(rot, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    rot = __builtin_rvtt_sfpshft2_subvec_shfl1(rot, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
    acc = __builtin_rvtt_sfpiadd_v(rot, acc, sfpi::SFPIADD_MOD1_CC_NONE);
    STROW(acc, 0); // per-row totals, all lanes of each row

    // (b) cross-row combine via SFPTRANSP: scatter the 4 row-totals of acc
    // across four registers, add them, then row-0 -> all rows via SFPCONFIG
    // vertical broadcast (dest 11).
    auto z = __builtin_rvtt_sfpxloadi(nullptr, 0, 0, 0, 31);
    auto z2 = __builtin_rvtt_sfpxloadi(nullptr, 0, 0, 0, 31);
    auto z3 = __builtin_rvtt_sfpxloadi(nullptr, 0, 0, 0, 31);
    auto t = __builtin_rvtt_sfptransp(acc, z, z2, z3);
    auto t0 = __builtin_rvtt_sfpselect4(t, 0);
    auto t1 = __builtin_rvtt_sfpselect4(t, 1);
    auto t2 = __builtin_rvtt_sfpselect4(t, 2);
    auto t3 = __builtin_rvtt_sfpselect4(t, 3);
    auto s = __builtin_rvtt_sfpiadd_v(t0, t1, sfpi::SFPIADD_MOD1_CC_NONE);
    s = __builtin_rvtt_sfpiadd_v(t2, s, sfpi::SFPIADD_MOD1_CC_NONE);
    s = __builtin_rvtt_sfpiadd_v(t3, s, sfpi::SFPIADD_MOD1_CC_NONE);
    __builtin_rvtt_sfpwriteconfig_v(s, 11);
    STROW(__builtin_rvtt_sfpreadlreg(11), 1);
}

// --- modes 16/17: register-axis sort4 network --------------------------------
inline void sort4_regs(bool sandwich)
{
    auto v0 = LDROW(0);
    auto v1 = LDROW(1);
    auto v2 = LDROW(2);
    auto v3 = LDROW(3);
    if (sandwich)
    {
        auto t = __builtin_rvtt_sfptransp(v0, v1, v2, v3);
        v0 = __builtin_rvtt_sfpselect4(t, 0);
        v1 = __builtin_rvtt_sfpselect4(t, 1);
        v2 = __builtin_rvtt_sfpselect4(t, 2);
        v3 = __builtin_rvtt_sfpselect4(t, 3);
    }
#define CE(a, b)                                          \
    do                                                    \
    {                                                     \
        auto r_ = __builtin_rvtt_sfpswap((a), (b), 1);    \
        (a) = __builtin_rvtt_sfpselect2(r_, 0);           \
        (b) = __builtin_rvtt_sfpselect2(r_, 1);           \
    } while (0)
    CE(v0, v1);
    CE(v2, v3);
    CE(v0, v2);
    CE(v1, v3);
    CE(v1, v2);
#undef CE
    if (sandwich)
    {
        auto t = __builtin_rvtt_sfptransp(v0, v1, v2, v3);
        v0 = __builtin_rvtt_sfpselect4(t, 0);
        v1 = __builtin_rvtt_sfpselect4(t, 1);
        v2 = __builtin_rvtt_sfpselect4(t, 2);
        v3 = __builtin_rvtt_sfpselect4(t, 3);
    }
    STROW(v0, 0);
    STROW(v1, 1);
    STROW(v2, 2);
    STROW(v3, 3);
}

// --- mode 18: partial LO16/HI16 companion roundtrip ---------------------------
inline void body_companion()
{
    // Pre-fill scratch output rows with a known 32b baseline (input row 2)
    auto base = LDROW(2);
    STROW(base, 4);
    STROW(base, 5);
    // Merge-load a packed companion from input rows 0 (lo16) and 1 (hi16):
    // first load leaves the high half from base's register value -- start
    // from a defined register (base) via the _lv load, DS/bridge pattern.
    auto c = __builtin_rvtt_sfpload(nullptr, 2 * 0, 0, 0, FMT_LO16_ONLY, NOINC);
    c = __builtin_rvtt_sfpload_lv(nullptr, c, 2 * 1, 0, 0, FMT_HI16_ONLY, NOINC);
    STROW(c, 0); // the packed word, 32b view
    // Partial stores onto the pre-filled baseline rows:
    __builtin_rvtt_sfpstore(nullptr, c, 192 + 2 * 4, 0, 0, FMT_LO16_ONLY, NOINC);
    __builtin_rvtt_sfpstore(nullptr, c, 192 + 2 * 5, 0, 0, FMT_HI16_ONLY, NOINC);
    // UINT16 store of the packed word to a zeroed row:
    auto z = __builtin_rvtt_sfpxloadi(nullptr, 0, 0, 0, 31);
    STROW(z, 6);
    __builtin_rvtt_sfpstore(nullptr, c, 192 + 2 * 6, 0, 0, FMT_U16, NOINC);
    // In-Dst roundtrip: store the packed word's halves to a scratch pair via
    // 16-bit stores, then merge-load them back and re-store as INT32.
    STROW(z, 7);
    STROW(z, 8);
    __builtin_rvtt_sfpstore(nullptr, c, 192 + 2 * 7, 0, 0, FMT_U16, NOINC);       // lo half
    __builtin_rvtt_sfpstore(nullptr, c, 192 + 2 * 8, 0, 0, FMT_HI16_ONLY, NOINC); // hi half
    auto rt = __builtin_rvtt_sfpload(nullptr, 192 + 2 * 7, 0, 0, FMT_LO16_ONLY, NOINC);
    rt = __builtin_rvtt_sfpload_lv(nullptr, rt, 192 + 2 * 8, 0, 0, FMT_HI16_ONLY, NOINC);
    STROW(rt, 9);
}

// --- mode 30: BF16 store onto known INT32 content (RMW probe, #2475) ----------
inline void body_bf16_rmw()
{
    // Baseline: fill output rows 0 and 1 with known 32b patterns.
    auto base0 = LDROW(0);
    auto base1 = LDROW(1);
    STROW(base0, 0);
    STROW(base1, 1);
    auto v = LDROW(3);
    // Arm 1 -- plain BF16-format store onto row 0's 32b content.  Three
    // candidate models for the paired low half: doc (SFPSTORE.md: only the
    // 16b cell written -> preserved), pinned sim (write_dst16b/write_dst32b:
    // ZEROED), silicon claim (tt-blaze #2475: BF16-canonicalized).
    __builtin_rvtt_sfpstore(nullptr, v, 192 + 2 * 0, 0, 0, FMT_BF16, NOINC);
    // Arm 2 -- the same store inside an ENABLE_DEST_INDEX window: the pinned
    // sim carries a low-half-PRESERVE special case for this state (tensix.cpp
    // sfpstore_values `lane_config & 4` arm, the TopK-motivated path flagged
    // by the hardcoding audit).  Pin it.
    TTI_SFPCONFIG(0x0004, 0xF, 1);
    __builtin_rvtt_sfpstore(nullptr, v, 192 + 2 * 1, 0, 0, FMT_BF16, NOINC);
    TTI_SFPCONFIG(0x0000, 0xF, 1);
}

inline void probe_body()
{
    if constexpr (PROBE_MODE == 0)
        body_identity();
    else if constexpr (PROBE_MODE == 1)
        body_rowtag();
    else if constexpr (PROBE_MODE == 2)
        body_lanetag();
    else if constexpr (PROBE_MODE == 3)
        body_transp8();
    else if constexpr (PROBE_MODE == 4)
        body_rot();
    else if constexpr (PROBE_MODE == 5)
        body_copy4_family<sfpi::SFPSHFT2_MOD1_COPY4>();
    else if constexpr (PROBE_MODE == 6)
        body_copy4_family<sfpi::SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4>();
    else if constexpr (PROBE_MODE == 7)
        body_copy4_family<sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4>();
    else if constexpr (PROBE_MODE == 8)
        body_swap_1234();
    else if constexpr (PROBE_MODE == 9)
        body_swap_5678();
    else if constexpr (PROBE_MODE == 10)
        body_swap_0_and_1();
    else if constexpr (PROBE_MODE == 11)
        body_exchange_flip();
    else if constexpr (PROBE_MODE == 12)
        body_lane_masked_flip();
    else if constexpr (PROBE_MODE == 13)
        body_indexed_swap();
    else if constexpr (PROBE_MODE == 14)
        body_config_broadcast();
    else if constexpr (PROBE_MODE == 15)
        body_reduce_int();
    else if constexpr (PROBE_MODE == 16)
        sort4_regs(false);
    else if constexpr (PROBE_MODE == 17)
        sort4_regs(true);
    else if constexpr (PROBE_MODE == 18)
        body_companion();
    else if constexpr (PROBE_MODE == 30)
        body_bf16_rmw();
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();
    math::reset_counters(p_setrwc::SET_ABD_F);
    _llk_math_welfords_sfpu_params_(+[]()
    {
        /* The raw probe bodies carry no predication, so the compiler
           emits no SFPENCC; enable all lanes explicitly.  */
        __builtin_rvtt_sfpencc_all_lanes();
        probe_body();
    }, 0);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_Res[2]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(3, L1_ADDRESS(params.buffer_Res[3]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
