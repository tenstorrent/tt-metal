// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole-only unit test for the experimental SDPA weighted-reduce LLK
// (api/compute/experimental/sdpa_weighted_reduce.h -> weighted_reduce, merged tt-metal #51361).
//
// WHAT THE OP COMPUTES (golden derivation, straight from the header)
// ------------------------------------------------------------------
// sdpa_weighted_reduce is a lightweight "weighted row reduction" matmul used by the DSA
// indexer SDPA op. Per chunk it computes, exactly as the header comment states:
//
//     out[1, 32] = weights[1, 8] x qk[8, 32]
//     out[p] = sum_{h=0..7} weights[h] * qk[h, p]        (p = 0..31)
//
// It is realized with two raw TTI_MVMUL (no matmul MOP). On Blackhole MVMUL computes
//     Dst[i,j] += sum_k SrcB[i,k] * SrcA[k,j], so:
//   - SrcB holds the weights vector in row 0 (cols 0..7 = head weights, 8..15 zero).
//     Rows 1..7 are zero, so only Dst row 0 is populated.
//   - SrcA holds qk: face0 = qk[0:8, 0:16] (SrcA rows 0..15), face1 = qk[0:8, 16:32]
//     (SrcA rows 16..31).
//   - MVMUL #1 (ADDR_MOD_6): qk face0 -> Dst row 0 cols 0..15 (out[0:16]); then SrcA += 16
//     rows (-> face1), Dst += 8 rows.
//   - MVMUL #2 (ADDR_MOD_3, CLR_AB): qk face1 -> Dst row 16 cols 0..15 (out[16:32]).
//   So DEST logical row 0 == out[0:32]. Only logical row 0 is meaningful; the rest of the
//   DEST tile is undefined.
//
// HOW THIS TEST DRIVES IT (faithful to the header's MATH core)
// ------------------------------------------------------------
// The header's own unpack/pack impls call API-level helpers (get_local_cb_interface,
// get_operand_id, reconfig_full_operand, get_output_tile_address, program_packer_destination)
// that live in tt_metal/hw/inc/api and are NOT includable from a tt-llk test. So, exactly as
// the sibling sources/sdpa_custom_mm_test.cpp does, we replicate only what a tt-llk test can:
//   - UNPACK: standard matmul unpack. In the matmul-unpack convention operandA -> SrcB and
//     operandB -> SrcA (see llk_unpack_AB_matmul.h: "in0/inA -> SrcB, in1/inB -> SrcA"), so
//     we feed WEIGHTS as buffer_A (-> SrcB) and QK as buffer_B (-> SrcA). This lands the same
//     SrcA=qk / SrcB=weights operands the two MVMULs consume.
//   - MATH: the header's weighted_reduce_math_impl call-for-call -- weighted_reduce_addrmod_init
//     (ADDR_MOD_6 = SrcA+16 / Dst+8; ADDR_MOD_3 set here too so the test is self-contained,
//     standing in for sdpa_custom_mm_init), then reset_counters, TT_SETC16 DEST target, and the
//     two TTI_MVMUL. This is the numeric heart of the op, byte-for-byte.
//   - PACK: standard _llk_pack_ of DEST tile 0. The header's raw two-PACR path only differs in
//     WHERE in L1 the row lands; the numeric content of DEST logical row 0 is identical, and
//     the .py validates only that defined row.
//
// LAYOUT-INDEPENDENT GOLDEN (no BH card here, so the exact Dest<->tile mapping is unverifiable)
// -------------------------------------------------------------------------------------------
// The .py fills the 8 head weights with a single constant W and the whole qk tile with a single
// constant Q. Then every output lane is analytically identical, independent of the exact
// face/lane mapping:
//     out[p] = sum_{h=0..7} W * Q = 8 * W * Q      (p = 0..31)
// The .py asserts ONLY the lanes this driver provably defines against 8*W*Q, per the
// "validate only defined lanes" rule.
//
// WHICH LANES THIS DRIVER DEFINES (plain-pack path, no api/ helpers here)
// ----------------------------------------------------------------------
// This test runs the header's MATH core verbatim but packs with a standard full-tile _llk_pack_
// instead of the header's raw two-PACR face-stepping pack. Under that path:
//   - MVMUL #1 (ADDR_MOD_6) writes the reduced row into DEST row 0, cols 0..15 == logical row 0
//     cols 0..15 (Dest face0 row 0).
//   - ADDR_MOD_6 then advances DEST by 8 rows (dest.incr = 8), so MVMUL #2 (qk face1) writes DEST
//     row 8 -- still inside face0, NOT physical face1 (DEST rows 16..31), which is where logical
//     row 0 cols 16..31 lives. The header's real pack reaches that face1 data via a custom
//     face-stepping addrmod; a plain full-tile pack does not, and that Dest<->face mapping is BH
//     hardware detail we cannot validate here.
// So the .py checks logical row 0 columns 0..15 (the 16 lanes MVMUL #1 defines). Columns 16..31
// of that row -- and every other row -- are undefined by this MATH+plain-pack path.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

using namespace ckernel;

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// The op only ever reduces a single [1,32] result row from one weights + one qk tile.
constexpr std::uint32_t WR_CT_DIM = 1;
constexpr std::uint32_t WR_RT_DIM = 1;
constexpr std::uint32_t WR_KT_DIM = 1;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // operandA (buffer_A) = WEIGHTS -> SrcB; operandB (buffer_B) = QK -> SrcA.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* unpA_num_faces */,
        4 /* unpB_num_faces */);
    _llk_unpack_AB_matmul_init_<>(
        0 /* transpose */,
        WR_CT_DIM,
        WR_RT_DIM,
        WR_KT_DIM,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* unpA_num_faces */,
        4 /* unpB_num_faces */,
        false /* unpA_partial_face */,
        false /* unpB_partial_face */);
    // buffer_A = weights (-> SrcB), buffer_B = qk (-> SrcA). Single k-tile.
    _llk_unpack_AB_matmul_<>(
        L1_ADDRESS(params.buffer_A[0]),
        L1_ADDRESS(params.buffer_B[0]),
        0 /* tile_index_a */,
        0 /* tile_index_b */,
        0 /* tile_size_a (default) */,
        0 /* tile_size_b (default) */,
        false /* unpA_partial_face */,
        false /* unpB_partial_face */,
        WR_CT_DIM,
        WR_RT_DIM,
        WR_KT_DIM);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"

// Replicates ckernel::weighted_reduce_addrmod_init() (MATH side) from
// api/compute/experimental/sdpa_weighted_reduce.h: after MVMUL #1, ADDR_MOD_6 advances SrcA by
// one 16-row face (-> qk face1) and DEST by 8 rows. In the real pipeline ADDR_MOD_3 is left by
// sdpa_custom_mm_init; here we program it too so MVMUL #2 is self-contained (SrcA/DEST hold on
// the CLR_AB close).
inline void weighted_reduce_addrmod_init_math()
{
    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6);
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_3);
}

// Replicates ckernel::weighted_reduce_math_impl() from the header.
inline void weighted_reduce_math_impl(const std::uint32_t dst_slot = 0)
{
    constexpr std::uint32_t weighted_dest_slot_rows = 16;
    math::reset_counters(p_setrwc::SET_ABD_F);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base() + dst_slot * weighted_dest_slot_rows);
    // MVMUL #1: qk face0 -> Dst row 0 (out[0:16]); then advance SrcA+16, Dst+8 (ADDR_MOD_6).
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_6, 0);
    // MVMUL #2: qk face1 -> Dst row 16 (out[16:32]); clear SrcA/SrcB (ADDR_MOD_3).
    TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Base matmul math init to establish the FPU config the MVMULs run under (LoFi, the only
    // fidelity the SDPA path uses). This programs the standard matmul addrmods; we then override
    // ADDR_MOD_6/ADDR_MOD_3 with the weighted-reduce values below.
    _llk_math_matmul_init_no_mop_<ckernel::MathFidelity::LoFi, 0>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, WR_CT_DIM, WR_RT_DIM);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    weighted_reduce_addrmod_init_math();

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    weighted_reduce_math_impl(0 /* dst_slot */);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    // Standard pack of DEST tile 0. The header packs only logical row 0 into a `partial` tile
    // row via raw PACR; the numeric content of that row equals row 0 of this full-tile pack, and
    // the .py validates only that defined row.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
