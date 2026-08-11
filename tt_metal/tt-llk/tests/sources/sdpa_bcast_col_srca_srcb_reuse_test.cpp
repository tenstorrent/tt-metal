// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-fork experimental LLK sdpa_bcast_col_srca_srcb_reuse (tt-metal#47554 / tt-blaze#1971),
// pending promotion. Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in
// FPU<->SFPU signalling cadence (orthogonal to this numerical golden).
//
// What the op does (tt-blaze#1971): the softmax scale / normalize step, as a *DEST-to-DEST* eltwise. Unlike the
// sibling srcb_reuse primitive, this variant reuses DEST for BOTH operands, so NOTHING it computes comes from an
// unpacked operand:
//   - Its MOP body is [MOVD2A, MOVD2A, ELWMUL] (the two MOVD2As are a replay buffer), so SrcA rows 0-7 are REFILLED
//     FROM DEST[dst_index] immediately before every ELWMUL, and every addrmod has .srca.incr = 0. An unpacked SrcA
//     tile is overwritten before it can be read -- feeding one makes the Elwmul(src_A, src_B_bcast) golden
//     unreachable, which is what this test used to assert against.
//   - Its ELWMUL carries CLR_NONE (vs CLR_A in the srcb-only variant) and the preamble waits on
//     STALLWAIT(WAIT_SFPU | SRCA_VLD | SRCB_VLD). Both dvalids are satisfied by the demo's
//     _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_() (sdpa.h:82) -- the SrcA+SrcB helper, NOT the SrcB-only one.
//   - The ELWMULs ACCUMULATE into DEST, and SrcA is a copy of DEST, so the op computes
//         DEST[dst] = DEST[dst] + DEST[dst] * broadcast_col(DEST[isrc])
//     i.e. an in-place scale-by-(1 + scale). This is exactly why the demo's SFPU side subtracts 1 from the scale it
//     produces (sdpa.h: "Without -1: bcast = prev * exp + prev  /  With -1: bcast = prev * exp"). Verified on p100a.
//   - Both DEST indices are RAW DEST ROW offsets (TT_SETC16 of DEST_TARGET_REG_CFG_MATH_Offset), not tile indices,
//     so they are 64 apart per 32x32 dest tile.
//
// Geometry: the MOP is two 8-row ELWMULs with dest.incr == 8, i.e. 16 CONTIGUOUS dest rows, and srcb.incr == 0 so
// both halves reuse the same 8 per-row scales. That is the demo's tile: an 8x32 logical tile packed into one 16x16
// DEST face ("Each tile is 8x32, which is the same as a full 16x16 face" -- sdpa.h:317), dest rows 0-7 holding
// logical columns 0-15 and rows 8-15 holding columns 16-31. The test therefore drives a single 16x16 face
// (num_faces == 1 on the unpack/pack side); the MATH mop still gets num_faces == 2, the only value its
// configure_mop LLK_ASSERT permits, which is where the "two 8-row chunks" come from.
//
// Modeled here:
//   1. Seed DEST[DST_ROW] with the operand tile X (buffer_A) and DEST[SRC_ROW] with the column source P (buffer_B),
//      both via ordinary A2D datacopies. SRC_ROW != DST_ROW so the golden is a genuine two-operand function.
//   2. _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_() injects the STALL_UNPACK + SrcB and SrcA SET_DVALIDs (ZEROSRC,
//      no real data) that the math preamble's STALLWAIT(WAIT_SFPU | SRCA_VLD | SRCB_VLD) waits on.
//   3. _llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_(SRC_ROW) MOVD2Bs DEST[SRC_ROW] rows 0-7 -> SrcB rows 0-7,
//      then _llk_math_sdpa_bcast_col_srca_srcb_reuse_ runs the SRCB_BCAST_COL MOP over DEST[DST_ROW]. The execute
//      clears SrcA+SrcB (SETRWC CLR_AB) at the end — this variant has NO separate postamble.
//
// The math thread posts semaphore::FPU_SFPU with no SFPU consumer; a bare post does not block (mirrors the sibling
// srcb_reuse test), so no fake SFPU handshake is needed and there is no MATH-waits-on-SFPU deadlock.
//
// Signalling-only divergence vs tt-blaze: the promoted BLAZE execute drops `bool fused_signalling` (template position
// 8), making the output_granularity loop unconditional; the promoted BLAZE init gains a trailing defaulted
// `bool skip_addrmod=false`. Neither touches numerics. On promotion, drop the `fused_signalling=true` template arg on
// the execute below (keeping output_granularity=1), leave the init call unchanged, and drop the -Wunused #pragma
// shims once the promoted header is warning-clean. See promotion_notes.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

static constexpr std::uint32_t NUM_TILES = 1;

// DEST tile indices for the two A2D seed datacopies (set_dst_write_addr scales these by 64 rows for Tile32x32):
//   tile 0 holds the operand X that the op scales in place, tile 1 holds the column source P.
static constexpr std::uint32_t DST_TILE = 0;
static constexpr std::uint32_t SRC_TILE = 1;

// ...and the same two locations as RAW DEST ROW offsets, which is what this primitive's preamble/execute take
// (TT_SETC16 of DEST_TARGET_REG_CFG_MATH_Offset, no tile-index scaling). SRC_ROW != DST_ROW is what makes the golden
// a real two-operand function instead of X * bcast_col(X).
static constexpr std::uint32_t DST_ROW = DST_TILE * 64;
static constexpr std::uint32_t SRC_ROW = SRC_TILE * 64;

// output_granularity gates how many tiles run between FPU_SFPU posts. Pin 1 to preserve per-tile FPU_SFPU cadence
// (matches the demo behavior). The golden ignores signalling cadence — this only affects when the (unconsumed) posts
// fire.
static constexpr std::uint32_t OUTPUT_GRANULARITY = 1;

// num_faces MUST be a compile-time constant on the math thread: sdpa_bcast_col_srca_srcb_reuse_configure_addrmod feeds
// the ADDR_MOD dest.incr through a SETC16 whose immediate takes the "n" (integer-constant) asm constraint. Passing
// params.num_faces (runtime) trips "impossible constraint in 'asm'". 2 is also the only value
// sdpa_bcast_col_srca_srcb_reuse_configure_mop's LLK_ASSERT permits; on the MATH side it is the mop's inner-loop
// count, i.e. the two 8-row ELWMUL chunks that together cover the tile's 16 dest rows (see the banner). It is
// deliberately NOT params.num_faces, which is 1 here -- the unpack/pack side sees a single 16x16 face.
// MATH_NUM_FACES comes from the MATH_NUM_FACES TemplateParameter in the generated build header, so the
// value lives in python only (see helpers/test_variant_parameters.py:MATH_NUM_FACES).

// This advance test exercises the MUL (softmax-scale) instantiation, LoFi fidelity.
static constexpr EltwiseBinaryType SDPA_OP  = EltwiseBinaryType::ELWMUL;
static constexpr MathFidelity SDPA_FIDELITY = MathFidelity::LoFi;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbols under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// On promotion, repoint the -I in test_config.py so this resolves to the canonical header and this line is unchanged.
//
// The unpromoted demo-fork header declares MOP-config locals (outerloop/innerloop) and takes unpack_src/dst_format
// params that the SCALAR/num_faces==2 path we instantiate does not read. The demo build tolerates these; the tt-llk
// harness compiles with -Werror -Wunused-variable -Wunused-parameter, so suppress the pre-existing warnings without
// editing the byte-identical shadow header. The offending locals live inside template bodies, so suppress at file
// scope (an include-only wrap does not reach the instantiation point). Remove on promotion once the canonical header
// is warning-clean.
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
    // One 16x16 face per tile: the op's 8x32 logical tile is packed into a single 16-row DEST face (see the banner).
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), 1 /* num_faces_r_dim */, 1 /* num_faces_c_dim */};

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        params.num_faces /* unpA_num_faces */,
        params.num_faces /* unpB_num_faces */);

    // Step 1: unpack the two DEST seeds. buffer_A carries X (the operand the op scales in place) and buffer_B carries
    // P (the column source); the MATH thread A2D-datacopies them into DEST[DST_TILE] and DEST[SRC_TILE].
    //
    // There is deliberately NO operand unpack into SrcA and no _llk_unpack_A_sdpa_init_ here: the MOP refills SrcA
    // from DEST with MOVD2A before every ELWMUL, so an unpacked SrcA tile would be discarded. The demo agrees --
    // sdpa_bcast_col_srca_srcb_reuse_tiles_init() touches the MATH thread only (sdpa.h:75-78).
    _llk_unpack_A_init_<BroadcastType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);

    // Step 2: satisfy the preamble's STALLWAIT(WAIT_SFPU | SRCA_VLD | SRCB_VLD) without pretending SrcA carries an
    // operand. This is the SrcA+SrcB helper (3 instructions), not the SrcB-only _llk_unpack_A_sdpa_set_srcb_dummy_valid_
    // (2 instructions) -- this variant's preamble waits on SRCA_VLD too, so the SrcB-only helper would leave MATH
    // stalled. The demo calls the same helper here (sdpa.h:82).
    _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_();
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// See the unpack-side note above: the unpromoted demo-fork header has pre-existing unused-variable declarations
// (addr_mod / innerloop / outerloop locals) that trip the harness's -Werror. Suppress at file scope (the offending
// vars live inside template bodies, so an include-only wrap does not reach the instantiation point). Remove on
// promotion once the canonical header is clean.
//
// The srca_srcb addrmod helper (sdpa_bcast_col_srca_srcb_reuse_configure_addrmod) additionally leaves its `num_faces`
// param unread on the LoFi ELWMUL path we instantiate (the dest.incr is a fixed 8 here, not derived from num_faces),
// so also suppress -Wunused-parameter on this thread. Remove on promotion once the canonical header is clean.
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_sdpa_bcast_col_srca_srcb_reuse.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Step 1: copy the two seeds (unpacked into SrcA by unpack step 1, in this order) into DEST. X goes to
    // DEST[DST_TILE] (the op scales it in place) and the column source P to DEST[SRC_TILE].
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(params.num_faces, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(DST_TILE, formats.math, formats.math, params.num_faces);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_TILE, formats.math, formats.math, params.num_faces);

    // Step 2: SDPA column-broadcast SrcA+SrcB-reuse eltwise.
    //   init     -> program addr_mods + the [MOVD2A, MOVD2A, ELWMUL] SRCB_BCAST_COL MOP.
    //   preamble -> STALLWAIT(WAIT_SFPU|SRCA_VLD|SRCB_VLD) on the unpacker's dummy SrcA+SrcB valids, point DEST at
    //               SRC_ROW, then MOVD2B DEST[SRC_ROW] rows 0-7 -> SrcB rows 0-7 (the per-row column scale).
    //   execute  -> for each of two 8-row chunks of DEST[DST_ROW]: MOVD2A those rows into SrcA, then accumulate
    //               SrcA * broadcast_col(SrcB) back onto them. Net effect
    //                   DEST[DST_ROW] = X + X * broadcast_col(P)   with X the seed at DST_ROW, P the seed at SRC_ROW.
    //               clear_dest stays false: X in DEST is an INPUT here, not stale state to be cleared.
    //               The trailing SETRWC CLR_AB releases SrcA/SrcB (this variant has no separate postamble).
    //
    // fused_signalling=true selects the output_granularity loop on the DEMO copy (per-tile FPU_SFPU with
    // OUTPUT_GRANULARITY=1). On promotion the BLAZE execute drops fused_signalling and the granularity loop is
    // unconditional — drop that template arg then; the golden is unaffected either way.
    _llk_math_sdpa_bcast_col_srca_srcb_reuse_init_<SDPA_OP, NUM_TILES, SDPA_FIDELITY>(MATH_NUM_FACES, 0 /* acc_to_dest */);
    _llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_<DST_SYNC, is_fp32_dest_acc_en, false /* clear_dest */>(SRC_ROW);
    _llk_math_sdpa_bcast_col_srca_srcb_reuse_<
        SDPA_OP,
        NUM_TILES,
        DST_SYNC,
        is_fp32_dest_acc_en,
        SDPA_FIDELITY,
        false /* clear_dest */,
        false /* skip_signalling */,
        true /* fused_signalling — DEMO-only; drop on promotion */,
        OUTPUT_GRANULARITY>(DST_ROW);

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
