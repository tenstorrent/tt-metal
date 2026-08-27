// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers experimental LLK sdpa_bcast_col_srcb_reuse (+ unpack_A_sdpa) (tt-metal#47554 /
// tt-blaze#1971), promoted into tt_llk_blackhole/llk_lib/experimental/ on main by #53295. The includes below resolve
// through the canonical -I; the demo-fork shadow tree this test was first written against no longer exists.
//
// What the op does (tt-blaze#1971): eltwise ADD/SUB/MUL of a per-tile operand (SrcA) with a *column* broadcast
// (SrcB) where the column source is DEST reused as a source register (DEST -> SrcB via MOVD2B), reused across every
// SrcA row (the softmax scale / normalize step). This advance test exercises the MUL (softmax-scale) instantiation,
// which has high-fidelity ELWMUL support; here we pin LoFi for the small bf16 grid.
//
// SrcB-reuse + set_srcb_dummy_valid handshake modeled here (same shape as transpose_dest_test.cpp's dest-reuse
// handshake; the external reference for the op itself is tt-blaze#1971, cited above):
//   1. Seed DEST[SRC_INDEX] with the column-source tile (buffer_B) via an ordinary A2D datacopy. In the real op this
//      tile is the per-column softmax scale produced by a prior reduce; here we feed it directly so the golden is a
//      plain column broadcast.
//   2. Call _llk_unpack_A_sdpa_set_srcb_dummy_valid_() -- BEFORE the operand unpacks -- to inject STALL_UNPACK plus a
//      UNPACR_NOP SET_DVALID on SrcB with no real data, so the downstream dual-source ELWMUL sees SrcB "ready". This
//      is the dummy SrcB the math preamble's STALLWAIT(SRCB_VLD) waits on before its MOVD2B reads DEST into SrcB.
//   3. Unpack TWO operand tiles (buffer_A[0], buffer_A[1]) into SrcA under the SrcA-only llk_unpack_A_sdpa MOP. The
//      execute runs the MOP twice and every ELWMUL carries CLR_A, so it retires 4 SrcA dvalids; one tile stalls MATH
//      forever. The demo pairs the same way (compute_kernel_api/sdpa.h:56-57).
//   4. _llk_math_sdpa_bcast_col_srcb_reuse_preamble_ MOVD2Bs DEST rows -> SrcB (P1 and P2), then
//      _llk_math_sdpa_bcast_col_srcb_reuse_ runs the SRCB_BCAST_COL eltwise MOP and the postamble clears SrcB:
//      DEST[DST_INDEX] = A0 * broadcast_col(P1) + A1 * broadcast_col(P2)   (the ELWMULs accumulate).
//
// This kernel covers unpack_A_sdpa as well, because exercising either primitive in isolation is impossible:
// unpack_A_sdpa is init/mop-config plus the dummy-SrcB-valid helper and has no execute of its own, and the math op
// cannot run without them. All three of its symbols are driven here: _llk_unpack_A_sdpa_init_ (the SrcA-only UNPACR
// MOP), the base llk_unpack_A execute under that MOP, and _llk_unpack_A_sdpa_set_srcb_dummy_valid_. The test that
// pinned unpack_A_sdpa by name (test_unpack_A_sdpa.py) is owned by #53361.
//
// Blackhole-only. The golden is verified on Blackhole silicon (p100a), not compile-green only.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// Single 8x32 tile: two 8x16 faces (num_faces == 2). The SDPA mop config LLK_ASSERTs num_faces == 2 (the init's
// assert allows 1/2/4, but sdpa_bcast_col_srcb_reuse_configure_mop hard-asserts == 2), so a 2-face tile is the only
// shape this primitive instantiates -- and 8 rows is the only face_r_dim that matches what it writes. Verified on
// p100a: the MOP is one ELWMUL per face and each ELWMUL covers 8 dest rows, so the op's output tile is 8x32. This is
// also the demo's tile ("Each tile is 8x32, which is the same as a full 16x16 face" -- sdpa.h:317).
//
// Two column sources are needed, not one: the preamble MOVD2Bs DEST rows 0-7 into SrcB rows 0-7 (P1) and DEST rows
// 64-71 -- the first 8 rows of the NEXT 32x32 dest tile -- into SrcB rows 8-15 (P2). So P1 lives in DEST[SRC_INDEX]
// and P2 in DEST[SRC_INDEX + 1]. Reuse-in-place: the output goes back to DEST[DST_INDEX] == DEST[SRC_INDEX], which
// is safe because both MOVD2Bs run before the first ELWMUL.
static constexpr std::uint32_t NUM_TILES = 1;
static constexpr std::uint32_t SRC_INDEX = 0;
static constexpr std::uint32_t DST_INDEX = 0;

// num_faces MUST be a compile-time constant on the math thread: sdpa_bcast_col_srcb_reuse_configure_addrmod feeds
// (16 + (num_dest_faces - num_faces)*16) into the ADDR_MOD dest.incr, which lands in a SETC16 whose immediate takes
// the "n" (integer-constant) asm constraint, so a runtime value trips "impossible constraint in 'asm'". That is why
// the python driver passes NUM_FACES as a TEMPLATE parameter rather than a runtime one: `num_faces` below is the
// constexpr the generated build header declares (helpers/sdpa_bcast_utils.py), so the value lives in python only and
// the unpack/pack sides read the same constant the math mop does.

// This advance test exercises the MUL (softmax-scale) instantiation, LoFi fidelity.
static constexpr EltwiseBinaryType SDPA_OP  = EltwiseBinaryType::ELWMUL;
static constexpr MathFidelity SDPA_FIDELITY = MathFidelity::LoFi;

// The MOP's ELWMULs ACCUMULATE into DEST (verified on p100a: with clear_dest == false the packed result is
// seed + A0*bcast_col(P1) + A1*bcast_col(P2), i.e. the column-source seed still sitting in DEST[DST_INDEX]
// is added in at column 0). clear_dest == true makes the preamble ZEROACC the dest half *after* its MOVD2Bs
// have already latched P1/P2 into SrcB, so the op reduces to the clean two-term product. The demo takes the
// same branch on its normalize path (sdpa.h: sdpa_bcast_col_reuse_preamble<normalize>()).
static constexpr bool CLEAR_DEST = true;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbols under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// Resolved from the promoted experimental/ copy (landed on main via #53295); the demo-fork shadow tree this test was
// originally written against is gone.
//
// The promoted header still takes transpose_of_faces / unpack_src_format / unpack_dst_format params that the
// num_faces == 2 path we instantiate does not read, and the harness compiles with -Werror -Wunused-parameter. The
// offending params are on template bodies, so an include-only push/pop does not reach the instantiation point --
// suppress at file scope. Drop this once the promoted header is warning-clean (tracked in #53295).
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_sdpa.h"
// Base unpack_A supplies the per-tile execute for both the DEST seed and the operand stream; llk_unpack_A_sdpa.h is
// init/mop-config + the dummy-SrcB-valid helper only.
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // 8x32 tile: one face-row (num_faces_r_dim == 1) of two 8x16 faces (num_faces_c_dim == 2).
    // The op writes 8 rows per face, so 8 is the natural face_r_dim -- see the SDPA_TILE_R_DIM note above.
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim), static_cast<std::uint8_t>(FACE_C_DIM), 1 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.in0_face_r_dim /* unpA_face_r_dim */,
        params.in0_face_r_dim /* unpB_face_r_dim */,
        num_faces /* unpA_num_faces */,
        num_faces /* unpB_num_faces */);

    // Step 1: seed the two column sources into DEST. Plain unpack_A -> SrcA, math datacopy A2D.
    // P1 goes to DEST[SRC_INDEX] (the preamble MOVD2Bs its rows 0-7 into SrcB rows 0-7) and P2 to
    // DEST[SRC_INDEX + 1] (the preamble's third/fourth MOVD2B read DEST rows 64-71, i.e. the first
    // 8 rows of the next 32x32 DEST tile, into SrcB rows 8-15).
    _llk_unpack_A_init_<BroadcastType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_B[1]), formats.unpack_A_src, formats.unpack_A_dst);

    // Step 2: SDPA SrcA-only unpack init programs the MOP; the base unpack_A execute then streams the operand tiles
    // into SrcA. set_srcb_dummy_valid injects the stall + SrcB SET_DVALID (no real data) that the math preamble
    // STALLWAIT(SRCB_VLD) waits on before it MOVD2Bs DEST into SrcB.
    _llk_unpack_A_sdpa_init_<NUM_TILES, BroadcastType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, params.in0_face_r_dim, num_faces, formats.unpack_A_src, formats.unpack_A_dst);

    // The dummy SrcB valid MUST be issued BEFORE the operand unpacks, matching the demo call order
    // (sdpa.h: sdpa_bcast_col_reuse_preamble() runs before sdpa_bcast_col_reuse_tiles()). Its leading
    // STALLWAIT(STALL_UNPACK, UNPACK) drains the unpacker, and the two operand unpacks below fill both
    // SrcA banks and then block until the math execute frees one. If the dummy valid came after them the
    // unpacker would be blocked on banks only the execute frees while math sat in the preamble's
    // STALLWAIT(SRCB_VLD) waiting for this instruction -- a deadlock.
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();

    // Two operand tiles, not one. _llk_math_sdpa_bcast_col_srcb_reuse_ runs the MOP twice (two consecutive
    // if-constexpr blocks, both taken for ELWMUL) and every ELWMUL carries CLR_A, so the execute retires
    // 2 * num_faces == 4 SrcA dvalids while one unpack of a 2-face tile supplies only 2. The demo pairs the
    // same way: compute_kernel_api/sdpa.h:56-57 issues two llk_unpack_A calls per math call, computing
    // cb_l1 * P1 + cb_l2 * P2. Supplying one tile stalls MATH forever on the second run.
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[1]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry), from the promoted
// experimental/ copy.
//
// Its configure_mop declares addr_mod / innerloop / outerloop (llk_math_sdpa_bcast_col_srcb_reuse.h:23-25) without
// reading them on the path we instantiate, which trips -Werror=unused-variable. Same file-scope suppression rationale
// as the unpack side: the declarations are inside template bodies, so an include-only push/pop does not reach the
// instantiation point. These are the symbols #53361 asked @pmilenkovicTT to drop in #53295; this shim goes when they do.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "experimental/llk_math_sdpa_bcast_col_srcb_reuse.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Step 1: copy the two column sources (unpacked into SrcA by unpack step 1) into DEST[SRC_INDEX] and
    // DEST[SRC_INDEX + 1]. The preamble's four MOVD2Bs read DEST rows 0-7 and 64-71, i.e. the first 8 rows
    // of each of these two 32x32 DEST tiles, into SrcB rows 0-7 (P1) and 8-15 (P2).
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(num_faces, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_INDEX, formats.math, formats.math, num_faces);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_INDEX + 1, formats.math, formats.math, num_faces);

    // Step 2: SDPA column-broadcast SrcB-reuse eltwise.
    //   init      -> program addr_mods + the SRCB_BCAST_COL MUL MOP.
    //   preamble  -> STALLWAIT(SRCB_VLD) on the unpacker's dummy SrcB valid, then MOVD2B DEST rows -> SrcB.
    //   execute   -> DEST[DST_INDEX] = SrcA(operand) * broadcast_col(scale).
    //   postamble -> SETRWC CLR_B (release the reused SrcB).
    _llk_math_sdpa_bcast_col_srcb_reuse_init_<SDPA_OP, NUM_TILES, SDPA_FIDELITY>(num_faces, 0 /* acc_to_dest */);
    _llk_math_sdpa_bcast_col_srcb_reuse_preamble_<DST_SYNC, is_fp32_dest_acc_en, CLEAR_DEST>();
    _llk_math_sdpa_bcast_col_srcb_reuse_<SDPA_OP, NUM_TILES, DST_SYNC, is_fp32_dest_acc_en, SDPA_FIDELITY, CLEAR_DEST>(DST_INDEX);
    _llk_math_sdpa_bcast_col_srcb_reuse_postamble_();

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Partial-face pack: the op's output tile is 8x32 (two 8x16 faces), and each face still occupies a full
    // 16-row DEST slot (the MOP's ADDR_MOD_0 steps dest by 16 between the two ELWMULs). Configuring the packer
    // with face_r_dim == in0_face_r_dim / num_faces == 2 / partial_face makes it read 8 rows out of each of the
    // two 16-row DEST faces. With the default full-tile config it instead reads DEST rows 0-15 as one face, so
    // the second half of the buffer comes back as the ZEROACC'd rows 8-15 rather than output columns 16-31.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, num_faces, true /* partial_face */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.in0_face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
