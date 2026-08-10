// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-tree experimental LLK sdpa_bcast_col_srcb_reuse (+ unpack_A_sdpa) (tt-metal#47554 /
// tt-blaze#1971), pending promotion into tt_llk_blackhole/llk_lib/experimental/. Include path below must be repointed
// on promotion. Primitives verified byte-identical to tt-blaze main as of this writing.
//
// What the op does (tt-blaze#1971): eltwise ADD/SUB/MUL of a per-tile operand (SrcA) with a *column* broadcast
// (SrcB) where the column source is DEST reused as a source register (DEST -> SrcB via MOVD2B), reused across every
// SrcA row (the softmax scale / normalize step). This advance test exercises the MUL (softmax-scale) instantiation,
// which has high-fidelity ELWMUL support; here we pin LoFi for the small bf16 grid.
//
// SrcB-reuse + set_srcb_dummy_valid handshake modeled here (mirrors rmsnorm_test.cpp's dest-reuse handshake):
//   1. Seed DEST[SRC_INDEX] with the column-source tile (buffer_B) via an ordinary A2D datacopy. In the real op this
//      tile is the per-column softmax scale produced by a prior reduce; here we feed it directly so the golden is a
//      plain column broadcast.
//   2. Unpack the operand tile (buffer_A) into SrcA using the SrcA-only llk_unpack_A_sdpa MOP, then call
//      _llk_unpack_A_sdpa_set_srcb_dummy_valid_(): it injects STALL_UNPACK + a UNPACR_NOP SET_DVALID on SrcB with no
//      real data, so the downstream dual-source ELWMUL sees SrcB "ready". This is the dummy SrcB the math preamble's
//      STALLWAIT(SRCB_VLD) waits on before its MOVD2B reads DEST into SrcB.
//   3. _llk_math_sdpa_bcast_col_srcb_reuse_preamble_ MOVD2Bs DEST rows -> SrcB (the column source), then
//      _llk_math_sdpa_bcast_col_srcb_reuse_ runs the SRCB_BCAST_COL eltwise MOP:
//      DEST[DST_INDEX] = SrcA(operand) * broadcast_col(scale), and the postamble clears SrcB.
//
// Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is pending
// Blackhole hardware/CI; this host is Wormhole.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// Single 16x32 "tiny" tile (num_faces == 2). The SDPA mop config LLK_ASSERTs num_faces == 2 (see header-vs-#1971
// note in the report: the init's assert allows 1/2/4, but sdpa_bcast_col_srcb_reuse_configure_mop hard-asserts == 2),
// so a 2-face tile is the only shape this primitive instantiates. The column source lives in DEST[SRC_INDEX]; the
// operand-combined result is written back to DEST[DST_INDEX]. Reuse-in-place: SRC and DST are the same tile (the
// column source is consumed into SrcB by the preamble MOVD2B before the op writes its output).
static constexpr std::uint32_t NUM_TILES = 1;
static constexpr std::uint32_t SRC_INDEX = 0;
static constexpr std::uint32_t DST_INDEX = 0;

// num_faces MUST be a compile-time constant on the math thread: sdpa_bcast_col_srcb_reuse_configure_addrmod feeds
// (16 + (num_dest_faces - num_faces)*16) into the ADDR_MOD dest.incr, which lands in a SETC16 whose immediate takes
// the "n" (integer-constant) asm constraint. Passing params.num_faces (runtime) trips "impossible constraint in
// 'asm'". A single 16x32 tiny tile is always 2 faces (the only shape the mop config accepts), so pin it here.
static constexpr std::uint32_t NUM_FACES_CT = 2;

// This advance test exercises the MUL (softmax-scale) instantiation, LoFi fidelity.
static constexpr EltwiseBinaryType SDPA_OP    = EltwiseBinaryType::ELWMUL;
static constexpr MathFidelity SDPA_FIDELITY   = MathFidelity::LoFi;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbols under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// On promotion, repoint the -I in test_config.py so this resolves to the canonical header and this line is unchanged.
//
// The unpromoted demo-tree header declares MOP-config locals (outerloop/innerloop) and takes unpack_src/dst_format
// params that the SCALAR/num_faces==2 path we instantiate does not read. The demo build tolerates these; the tt-llk
// harness compiles with -Werror -Wunused-variable -Wunused-parameter, so suppress the pre-existing warnings without
// editing the byte-identical shadow header. The offending locals live inside template bodies, so suppress at file
// scope (an include-only wrap does not reach the instantiation point). Remove on promotion once the canonical header
// is warning-clean (see report: header-vs-#1971 note).
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
    // 16x32 tiny tile => num_faces == 2 (one face-row of two faces).
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), 1, 2 /* num_faces == 2 */};

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        params.num_faces /* unpA_num_faces */,
        params.num_faces /* unpB_num_faces */);

    // Step 1: seed DEST[SRC_INDEX] with the column-source tile (buffer_B). Plain unpack_A -> SrcA, math datacopy A2D.
    _llk_unpack_A_init_<BroadcastType::NONE>(0, 0, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);

    // Step 2: SDPA SrcA-only unpack init programs the MOP; the base unpack_A execute then streams the operand tile
    // into SrcA. set_srcb_dummy_valid then injects the stall + SrcB SET_DVALID (no real data) that the math preamble
    // STALLWAIT(SRCB_VLD) waits on before it MOVD2Bs DEST into SrcB.
    _llk_unpack_A_sdpa_init_<NUM_TILES, BroadcastType::NONE>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        FACE_R_DIM,
        params.num_faces,
        formats.unpack_A_src,
        formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// See the unpack-side note above: the unpromoted demo-tree header has pre-existing unused-variable declarations
// (addr_mod / innerloop / outerloop locals) that trip the harness's -Werror. Suppress at file scope (the offending
// vars live inside template bodies, so an include-only wrap does not reach the instantiation point). Remove on
// promotion once the canonical header is clean.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "llk_math_sdpa_bcast_col_srcb_reuse.h"
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

    // Step 1: copy the column-source tile (unpacked into SrcA by unpack step 1) into DEST[SRC_INDEX].
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(
        params.num_faces, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(
        SRC_INDEX, formats.math, formats.math, params.num_faces);

    // Step 2: SDPA column-broadcast SrcB-reuse eltwise.
    //   init      -> program addr_mods + the SRCB_BCAST_COL MUL MOP.
    //   preamble  -> STALLWAIT(SRCB_VLD) on the unpacker's dummy SrcB valid, then MOVD2B DEST rows -> SrcB.
    //   execute   -> DEST[DST_INDEX] = SrcA(operand) * broadcast_col(scale).
    //   postamble -> SETRWC CLR_B (release the reused SrcB).
    _llk_math_sdpa_bcast_col_srcb_reuse_init_<SDPA_OP, NUM_TILES, SDPA_FIDELITY>(
        NUM_FACES_CT, 0 /* acc_to_dest */);
    _llk_math_sdpa_bcast_col_srcb_reuse_preamble_<DST_SYNC, is_fp32_dest_acc_en, false /* clear_dest */>();
    _llk_math_sdpa_bcast_col_srcb_reuse_<
        SDPA_OP,
        NUM_TILES,
        DST_SYNC,
        is_fp32_dest_acc_en,
        SDPA_FIDELITY,
        false /* clear_dest */>(DST_INDEX);
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
