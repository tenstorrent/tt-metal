// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-fork experimental LLK unpack_A_sdpa (tt-metal#47554 / tt-blaze#1971), pending promotion.
// Include path (shadow -I) repoint on promotion. Primitive verified vs tt-blaze main as of this writing.
//
// This test is pinned to the DEMO-fork primitive header
// (models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_A_sdpa.h) via the shadow -I
// already registered in test_config.py, so it does not depend on the tt-blaze migration. The comparison phase found
// the demo-fork header byte-identical to tt-blaze main except for the copyright-holder comment (USA, Inc. vs AI ULC),
// so the numerics validated here match the canonical primitive.
//
// unpack_A_sdpa is init/mop-config + a dummy-SrcB-valid helper only; it has no per-tile execute of its own. This test
// drives all three of its symbols:
//   - _llk_unpack_A_sdpa_init_<num_tiles, BType>(...)      : programs the SrcA-only UNPACR MOP.
//   - the base llk_unpack_A execute                        : streams the operand tile into SrcA under that MOP.
//   - _llk_unpack_A_sdpa_set_srcb_dummy_valid_()           : injects STALL_UNPACK + a UNPACR_NOP SET_DVALID on SrcB
//                                                            (ZEROSRC, no real data) so the downstream dual-source
//                                                            eltwise's math preamble STALLWAIT(SRCB_VLD) does not
//                                                            stall. This is unpacker-side self-satisfied: no
//                                                            MATH-waits-on-SFPU handshake, so an isolated kernel does
//                                                            not deadlock and nothing needs faking.
//
// To exercise unpack_A_sdpa with a validatable NUMERIC golden, we pair it with the demo-fork math SDPA column-
// broadcast SrcB-reuse op (llk_math_sdpa_bcast_col_srcb_reuse.h): the column source is seeded into DEST via a plain
// A2D datacopy, MOVD2B'd into SrcB by the math preamble (which waits on the dummy SrcB valid this unpack helper
// injects), then multiplied against the SrcA operand. The golden is a plain column-broadcast MUL.
//
// Match the SrcA-only face/tile geometry: a single 8x32 tile (two 8x16 faces, num_faces == 2) -- the SDPA mop config
// accepts only a 2-face tile, and 8 rows is what its ELWMULs write. Two operand tiles are unpacked per math call,
// because the execute runs the MOP twice and every ELWMUL carries CLR_A.
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

// Single 8x32 tile: two 8x16 faces (num_faces == 2). The SDPA mop config accepts only a 2-face tile, and the MOP
// writes 8 dest rows per face, so 8x32 is the shape the op actually produces (the demo's tile too). Two column
// sources are needed: the math preamble MOVD2Bs DEST rows 0-7 into SrcB rows 0-7 (P1) and DEST rows 64-71 -- the top
// of the NEXT 32x32 dest tile -- into SrcB rows 8-15 (P2). So P1 is DEST[SRC_INDEX] and P2 is DEST[SRC_INDEX + 1].
// The result goes back to DEST[DST_INDEX] == DEST[SRC_INDEX]; that is safe because both MOVD2Bs run first.
static constexpr std::uint32_t NUM_TILES = 1;
static constexpr std::uint32_t SRC_INDEX = 0;
static constexpr std::uint32_t DST_INDEX = 0;

// num_faces MUST be a compile-time constant on the math thread: the SDPA addrmod config feeds an integer into a
// SETC16 whose immediate takes the "n" (integer-constant) asm constraint. The 8x32 tile is always 2 faces.
// MATH_NUM_FACES comes from the MATH_NUM_FACES TemplateParameter in the generated build header, so the
// value lives in python only (see helpers/test_variant_parameters.py:MATH_NUM_FACES).

// MUL (softmax-scale) instantiation, LoFi fidelity for the small bf16 grid.
static constexpr EltwiseBinaryType SDPA_OP  = EltwiseBinaryType::ELWMUL;
static constexpr MathFidelity SDPA_FIDELITY = MathFidelity::LoFi;

// The MOP's ELWMULs accumulate into DEST, so clear_dest == true is required: it makes the preamble ZEROACC the dest
// half AFTER its MOVD2Bs have latched P1/P2 into SrcB, leaving the clean two-term product rather than
// seed + A0*bcast_col(P1) + A1*bcast_col(P2). Matches the demo's normalize path.
static constexpr bool CLEAR_DEST = true;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// On promotion, repoint the -I in test_config.py so this resolves to the canonical header and this line is unchanged.
//
// The unpromoted demo-fork header declares MOP-config locals (outerloop/innerloop) and takes unpack_src/dst_format
// params that the num_faces==2 path we instantiate does not read. The demo build tolerates these; the tt-llk harness
// compiles with -Werror -Wunused-variable -Wunused-parameter, so suppress the pre-existing warnings without editing
// the byte-identical shadow header. The offending locals live inside template bodies, so suppress at file scope (an
// include-only wrap does not reach the instantiation point). Remove on promotion once the canonical header is
// warning-clean.
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llk_unpack_A_sdpa.h"
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
        params.num_faces /* unpA_num_faces */,
        params.num_faces /* unpB_num_faces */);

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
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, params.in0_face_r_dim, params.num_faces, formats.unpack_A_src, formats.unpack_A_dst);

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

// PRIMITIVE symbols under test. See the unpack-side note above: the unpromoted demo-fork header has pre-existing
// unused-variable declarations (addr_mod / innerloop / outerloop locals) that trip the harness's -Werror. Suppress at
// file scope (the offending vars live inside template bodies, so an include-only wrap does not reach the instantiation
// point). Remove on promotion once the canonical header is clean.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_sdpa_bcast_col_srcb_reuse.h"
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
    // DEST[SRC_INDEX + 1] -- the two tiles the preamble's four MOVD2Bs read (rows 0-7 and 64-71).
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(params.num_faces, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_INDEX, formats.math, formats.math, params.num_faces);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_INDEX + 1, formats.math, formats.math, params.num_faces);

    // Step 2: SDPA column-broadcast SrcB-reuse eltwise. The preamble's STALLWAIT(SRCB_VLD) is satisfied by the
    // unpacker's _llk_unpack_A_sdpa_set_srcb_dummy_valid_() (unpacker-side self-satisfied handshake), so this isolated
    // kernel does not deadlock.
    //   init      -> program addr_mods + the SRCB_BCAST_COL MUL MOP.
    //   preamble  -> STALLWAIT(SRCB_VLD) on the unpacker's dummy SrcB valid, then MOVD2B DEST rows -> SrcB.
    //   execute   -> DEST[DST_INDEX] = SrcA(operand) * broadcast_col(scale).
    //   postamble -> SETRWC CLR_B (release the reused SrcB).
    _llk_math_sdpa_bcast_col_srcb_reuse_init_<SDPA_OP, NUM_TILES, SDPA_FIDELITY>(MATH_NUM_FACES, 0 /* acc_to_dest */);
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
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true /* partial_face */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.in0_face_r_dim, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
