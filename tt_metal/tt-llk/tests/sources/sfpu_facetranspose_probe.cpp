// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X6 FPU face-transpose arsenal probe (lane FV, 2026-08-22).
//
// Structure mirrors sfpu_crosslane_probe.cpp (lane FB): UInt32
// unpack-to-dest input, PROBE_MODE if-constexpr dispatch, results stored
// to output tile 3 (and tile-2 region reads for the Dst-row calibration).
// The python side (test_crosslane_facetranspose.py) calibrates the
// tensor<->(row,lane) mapping empirically, derives the SFPU-view <->
// Dst16-row map from the Dst-row roundtrip mode, and compares EVERY lane
// against helpers/facetranspose_oracle.py (transcribed from the
// tt-isa-documentation MOVD2B/MOVB2A/MOVB2D/MOVA2D/TRNSPSRCB models).
//
// PROBE_MODE map (keep in sync with the python module):
//   0 identity      : out rows 0..15 = in rows 0..15         (calibration)
//   1 rowtag        : out row i = 0x00A00000 + i             (calibration)
//   2 lanetag       : out rows 0..15 = vConstTileId          (calibration)
//   3 dstrow cal    : per Dst16 row r in 0..15: MOVD2B(TF32, 1 row) ->
//                     MOVB2D back into Dst rows 32+r (face-2 area); out
//                     rows 0..7 = vector rows 16..23 (the face-2 view)
//   4 face 0        : sfpi::face_transpose_dst_32b<0> under the cfg
//                     block; out rows 0..15 = vector rows 0..15
//                     (transposed face 0 + untouched face 1 control)
//   5 face 1        : sfpi::face_transpose_dst_32b<16> (offset
//                     genericity); out rows 0..15 = vector rows 0..15
//   6 batch<2>      : sfpi::face_transpose_dst_32b_batch<2> (faces 0+1)
//   7 hi-stage      : passes 1+2 only (raw spelling of the surface
//                     words); out rows 0..7 = the hi16-transposed face
//                     (adjudicates the implied-SrcBFmt masking arm)
//   8 zeroflag twin : full choreography WITHOUT the zero-flag arm of the
//                     cfg block (contract-necessity: the oracle predicts
//                     the flushed lanes)

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
    // Grant the SrcA and SrcB banks to the math thread for the FPU
    // choreography (the transpose_dest_test.cpp cross-thread protocol;
    // one grant per math epoch).
    _llk_unpack_set_srcb_dummy_valid_();
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"
using namespace ckernel;

// Drop the sfpi arity-reducing macros; probe bodies use the raw builtin
// arity with a nullptr iptr (lane DS's proven pattern).
#ifdef __builtin_rvtt_sfpload
#undef __builtin_rvtt_sfpload
#endif
#ifdef __builtin_rvtt_sfpstore
#undef __builtin_rvtt_sfpstore
#endif
#ifdef __builtin_rvtt_sfpxloadi
#undef __builtin_rvtt_sfpxloadi
#endif

// ---------------------------------------------------------------------
// The charter's drift belt: the X6 surface carries its BH config-field
// constants as named constexprs; every one is cross-checked here against
// the production headers at compile time.
static_assert(sfpi::facetranspose_impl_::bh_alu_format_srca_addr32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, "X6 constant drift: SrcA fmt addr32");
static_assert(sfpi::facetranspose_impl_::bh_alu_format_srca_shamt == ALU_FORMAT_SPEC_REG0_SrcA_SHAMT, "X6 constant drift: SrcA fmt shamt");
static_assert(sfpi::facetranspose_impl_::bh_alu_format_srca_mask == ALU_FORMAT_SPEC_REG0_SrcA_MASK, "X6 constant drift: SrcA fmt mask");
static_assert(sfpi::facetranspose_impl_::bh_alu_fp32_enabled_addr32 == ALU_ACC_CTRL_Fp32_enabled_ADDR32, "X6 constant drift: Fp32 addr32");
static_assert(sfpi::facetranspose_impl_::bh_alu_fp32_enabled_shamt == ALU_ACC_CTRL_Fp32_enabled_SHAMT, "X6 constant drift: Fp32 shamt");
static_assert(sfpi::facetranspose_impl_::bh_alu_fp32_enabled_mask == ALU_ACC_CTRL_Fp32_enabled_MASK, "X6 constant drift: Fp32 mask");
static_assert(sfpi::facetranspose_impl_::bh_alu_zero_flag_dis_src_addr32 == ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32, "X6 constant drift: zero-flag addr32");
static_assert(sfpi::facetranspose_impl_::bh_alu_zero_flag_dis_src_shamt == ALU_ACC_CTRL_Zero_Flag_disabled_src_SHAMT, "X6 constant drift: zero-flag shamt");
static_assert(sfpi::facetranspose_impl_::bh_alu_zero_flag_dis_src_mask == ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK, "X6 constant drift: zero-flag mask");
static_assert(
    sfpi::facetranspose_impl_::bh_disable_implied_srca_fmt_setc16 == DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, "X6 constant drift: implied-fmt SETC16 index");
static_assert(sfpi::facetranspose_impl_::bh_fmt_float32 == static_cast<unsigned>(DataFormat::Float32), "X6 constant drift: Float32 code");
static_assert(sfpi::facetranspose_impl_::bh_fmt_tf32 == static_cast<unsigned>(DataFormat::Tf32), "X6 constant drift: Tf32 code");
static_assert(sfpi::facetranspose_impl_::bh_fmt_float16_b == static_cast<unsigned>(DataFormat::Float16_b), "X6 constant drift: Float16_b code");
static_assert(sfpi::facetranspose_impl_::bh_stall_cfg == p_stall::STALL_CFG, "X6 constant drift: STALL_CFG");
static_assert(sfpi::facetranspose_impl_::bh_wait_sfpu == p_stall::WAIT_SFPU, "X6 constant drift: WAIT_SFPU");
static_assert(sfpi::facetranspose_impl_::bh_srca_vld == p_stall::SRCA_VLD, "X6 constant drift: SRCA_VLD");
static_assert(sfpi::facetranspose_impl_::bh_srcb_vld == p_stall::SRCB_VLD, "X6 constant drift: SRCB_VLD");
static_assert(sfpi::facetranspose_impl_::bh_mov_dest_norm == p_mov::DEST_NORM, "X6 constant drift: DEST_NORM");
static_assert(sfpi::facetranspose_impl_::bh_mov_dest_32b_low == p_mov::DEST_32B_LOW, "X6 constant drift: DEST_32B_LOW");
static_assert(sfpi::facetranspose_impl_::bh_movd2b_mov_4_rows == p_movd2b::MOV_4_ROWS, "X6 constant drift: movd2b MOV_4_ROWS");
static_assert(sfpi::facetranspose_impl_::bh_movb2a_mov_4_rows == p_movb2a::MOV_4_ROWS, "X6 constant drift: movb2a MOV_4_ROWS");
static_assert(sfpi::facetranspose_impl_::bh_movb2d_mov_4_rows == p_movb2d::MOV_4_ROWS, "X6 constant drift: movb2d MOV_4_ROWS");
static_assert(sfpi::facetranspose_impl_::bh_mova2d_mov_8_rows == p_mova2d::MOV_8_ROWS, "X6 constant drift: mova2d MOV_8_ROWS");
static_assert(sfpi::facetranspose_impl_::bh_addr_mod_sfpu == ADDR_MOD_7, "X6 constant drift: SFPU addr mod");

namespace
{

// INT32 raw Dst access, NOINC addressing (lane DS constants): input
// vector rows at addr 2*i, output tile 3 at 192 + 2*i.
constexpr unsigned FMT_I32 = 4;
constexpr unsigned NOINC   = 7;

#define LDROW(i)    __builtin_rvtt_sfpload(nullptr, 2 * (i), 0, 0, FMT_I32, NOINC)
#define STROW(v, i) __builtin_rvtt_sfpstore(nullptr, (v), 192 + 2 * (i), 0, 0, FMT_I32, NOINC)

inline void copy_rows_out(unsigned src_vrow_base, unsigned n)
{
    for (unsigned i = 0; i < n; ++i)
    {
        STROW(LDROW(src_vrow_base + i), i);
    }
}

// Every mode must CONSUME the unpack thread's bank grants before the
// kernel ends: the harness keeps ONE sim device per pytest session, so an
// unconsumed dummy dvalid PERSISTS into the next test's kernel and wedges
// its unpacker (the cross-launch state-persistence class lane FS proved
// on silicon for replay state; same shape here for bank valids).  The
// choreography modes consume them naturally; the calibration modes drain
// them explicitly.
inline void drain_bank_grants()
{
    namespace fi = sfpi::facetranspose_impl_;
    __builtin_rvtt_ttstallwait(0x40 /* STALL_MATH */, fi::bh_srca_vld | fi::bh_srcb_vld);
    sfpi::face_transpose_release_banks();
}

inline void body_identity()
{
    copy_rows_out(0, 16);
    drain_bank_grants();
}

inline void body_rowtag()
{
#define ROWTAG(i) STROW(__builtin_rvtt_sfpxloadi(nullptr, 0x00A00000 + (i), 0, 0, 31), i)
    ROWTAG(0);
    ROWTAG(1);
    ROWTAG(2);
    ROWTAG(3);
    ROWTAG(4);
    ROWTAG(5);
    ROWTAG(6);
    ROWTAG(7);
    ROWTAG(8);
    ROWTAG(9);
    ROWTAG(10);
    ROWTAG(11);
    ROWTAG(12);
    ROWTAG(13);
    ROWTAG(14);
    ROWTAG(15);
#undef ROWTAG
    drain_bank_grants();
}

inline void body_lanetag()
{
    auto v = __builtin_rvtt_sfpreadlreg(15); /* vConstTileId */
    for (unsigned i = 0; i < 16; ++i)
    {
        STROW(v, i);
    }
    drain_bank_grants();
}

// --- mode 3: Dst16-row calibration ------------------------------------
// One row at a time through SrcB and back into the face-2 area
// (Dst16 rows 32+r).  SrcAFmt = Tf32 for the hi16-exact roundtrip; the
// stimulus keeps its tags in the hi16 so the SrcBFmt masking ambiguity
// cannot touch them.
inline void body_dstrow_cal()
{
    namespace fi = sfpi::facetranspose_impl_;
    sfpi::face_transpose_cfg_enter();
    fi::set_srca_format<fi::bh_fmt_tf32>();
#define RT1(r)                                 \
    __builtin_rvtt_ttmovd2b(0, 16, 7, 0, (r)); \
    __builtin_rvtt_ttmovb2d(0, 16, 7, 0, 32 + (r))
    RT1(0);
    RT1(1);
    RT1(2);
    RT1(3);
    RT1(4);
    RT1(5);
    RT1(6);
    RT1(7);
    RT1(8);
    RT1(9);
    RT1(10);
    RT1(11);
    RT1(12);
    RT1(13);
    RT1(14);
    RT1(15);
#undef RT1
    sfpi::face_transpose_cfg_leave();
    copy_rows_out(16, 8);
    sfpi::face_transpose_release_banks();
}

// --- modes 4/5: single-face surface transpose --------------------------
template <unsigned FaceRow>
inline void body_face()
{
    sfpi::face_transpose_cfg_enter();
    sfpi::face_transpose_dst_32b<FaceRow>();
    sfpi::face_transpose_cfg_leave();
    copy_rows_out(0, 16);
    sfpi::face_transpose_release_banks();
}

// --- mode 6: batched surface transpose ---------------------------------
inline void body_batch2()
{
    sfpi::face_transpose_dst_32b_batch<2>();
    copy_rows_out(0, 16);
    sfpi::face_transpose_release_banks();
}

// Raw spelling of the surface's per-pass words (stage/twin probes own
// their exact truncation points; kept textually parallel to
// face_transpose_dst_32b so a drift is a review diff).
inline void raw_pass1_lo16_park()
{
    namespace fi = sfpi::facetranspose_impl_;
    fi::set_srca_format<fi::bh_fmt_float16_b>();
    __builtin_rvtt_ttmovd2b(1, 16, 7, 2, 0);
    __builtin_rvtt_ttmovd2b(1, 20, 7, 2, 4);
    __builtin_rvtt_ttmovd2b(1, 24, 7, 2, 8);
    __builtin_rvtt_ttmovd2b(1, 28, 7, 2, 12);
    __builtin_rvtt_tttrnspsrcb();
    __builtin_rvtt_ttmovb2a(0, 7, 2, 16);
    __builtin_rvtt_ttmovb2a(4, 7, 2, 20);
    __builtin_rvtt_ttmovb2a(8, 7, 2, 24);
    __builtin_rvtt_ttmovb2a(12, 7, 2, 28);
}

inline void raw_pass2_hi16()
{
    namespace fi = sfpi::facetranspose_impl_;
    fi::set_srca_format<fi::bh_fmt_tf32>();
    __builtin_rvtt_ttmovd2b(0, 16, 7, 2, 0);
    __builtin_rvtt_ttmovd2b(0, 20, 7, 2, 4);
    __builtin_rvtt_ttmovd2b(0, 24, 7, 2, 8);
    __builtin_rvtt_ttmovd2b(0, 28, 7, 2, 12);
    __builtin_rvtt_tttrnspsrcb();
    __builtin_rvtt_ttmovb2d(0, 16, 7, 4, 0);
    __builtin_rvtt_ttmovb2d(0, 20, 7, 4, 4);
    __builtin_rvtt_ttmovb2d(0, 24, 7, 4, 8);
    __builtin_rvtt_ttmovb2d(0, 28, 7, 4, 12);
}

inline void raw_pass3_lo16_writeback()
{
    namespace fi = sfpi::facetranspose_impl_;
    fi::set_fp32_enabled<0>();
    fi::set_srca_format<fi::bh_fmt_float32>();
    __builtin_rvtt_ttmova2d(1, 0, 7, 2, 0);
    __builtin_rvtt_ttmova2d(1, 8, 7, 2, 8);
    fi::set_fp32_enabled<1>();
}

// --- mode 7: hi-stage truncation (passes 1+2 only) ----------------------
inline void body_hi_stage()
{
    sfpi::face_transpose_cfg_enter();
    raw_pass1_lo16_park();
    raw_pass2_hi16();
    sfpi::face_transpose_cfg_leave();
    copy_rows_out(0, 8);
    sfpi::face_transpose_release_banks();
}

// --- mode 8: zero-flag-off contract-necessity twin ----------------------
inline void body_zeroflag_twin()
{
    namespace fi = sfpi::facetranspose_impl_;
    // Deliberately NOT the cfg block: implied-format off + stall, and the
    // zero-flag FORCED to 0 (flush ENABLED) -- the contract-necessity
    // twin the oracle predicts flushed lanes for.
    __builtin_rvtt_ttsetc16(fi::bh_disable_implied_srca_fmt_setc16, 1);
    fi::cfg_field_rmw<fi::bh_alu_zero_flag_dis_src_addr32, fi::bh_alu_zero_flag_dis_src_shamt, fi::bh_alu_zero_flag_dis_src_mask, 0>();
    __builtin_rvtt_ttstallwait(fi::bh_stall_cfg, fi::bh_wait_sfpu | fi::bh_srca_vld | fi::bh_srcb_vld);
    raw_pass1_lo16_park();
    raw_pass2_hi16();
    raw_pass3_lo16_writeback();
    __builtin_rvtt_ttsetc16(fi::bh_disable_implied_srca_fmt_setc16, 0);
    copy_rows_out(0, 8);
    sfpi::face_transpose_release_banks();
}

inline void probe_body()
{
    if constexpr (PROBE_MODE == 0)
    {
        body_identity();
    }
    else if constexpr (PROBE_MODE == 1)
    {
        body_rowtag();
    }
    else if constexpr (PROBE_MODE == 2)
    {
        body_lanetag();
    }
    else if constexpr (PROBE_MODE == 3)
    {
        body_dstrow_cal();
    }
    else if constexpr (PROBE_MODE == 4)
    {
        body_face<0>();
    }
    else if constexpr (PROBE_MODE == 5)
    {
        body_face<16>();
    }
    else if constexpr (PROBE_MODE == 6)
    {
        body_batch2();
    }
    else if constexpr (PROBE_MODE == 7)
    {
        body_hi_stage();
    }
    else if constexpr (PROBE_MODE == 8)
    {
        body_zeroflag_twin();
    }
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
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();
    math::reset_counters(p_setrwc::SET_ABD_F);
    _llk_math_welfords_sfpu_params_(
        +[]()
        {
            __builtin_rvtt_sfpencc_all_lanes();
            probe_body();
        },
        0);
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
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
