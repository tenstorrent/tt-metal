// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-fork experimental LLK sdpa_custom_mm_reuse_dest_srcb (tt-metal#47554 / tt-blaze#1971),
// pending promotion. Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU
// signalling cadence (orthogonal to this numerical golden).
//
// The primitive under test is the demo-fork sdpa_custom_mm_reuse_dest_srcb matmul: a K-reduction custom matmul that
// UNPACKS ONLY SrcA (in1, the full [32,32] K-tiles) and REUSES SrcB FROM DEST (in0, the [1,32] partial tile), moving
// DEST rows into SrcB via MOVD2B each K-iteration. Contract: ct_dim==1, rt_dim==1, single output tile shape [1,32],
// kt_dim even (2..256), nt_dim 1..16. LoFi-only (math init takes no MathFidelity template on the demo call path; the
// demo-fork primitive templates MathFidelity, which we pin to LoFi).
//
// dest<-SFPU handshake fake (mandatory to avoid deadlock on Blackhole hardware):
//   The MATH primitive does, at the TOP of every K-iteration,
//       t6_semaphore_wait_on_zero<STALL_MATH>(SFPU_FPU)     // SFPU_FPU == semaphore::UNPACK_MATH_DONE (index 6)
//   (stall while the semaphore is zero, i.e. wait until the "SFPU" side POSTs it), then MOVD2Bs DEST->SrcB, then
//       t6_semaphore_get<MATH>(SFPU_FPU)                     // decrement back to zero
//   In the demo this semaphore is posted by a paired SFPU kernel; in this isolated compute-only kernel there is no
//   SFPU op, so MATH would block forever on the wait_on_zero. We fake the SFPU side on the UNPACK thread (Tensix
//   semaphores are shared across all three threads): after its unpack work, UNPACK posts semaphore::UNPACK_MATH_DONE
//   exactly kt_dim times. MATH's per-K wait_on_zero then sees non-zero and proceeds, and MATH's per-K get decrements
//   it, so posts (kt_dim) == gets (kt_dim) and nothing leaks. kt_dim <= 4 here, well under SEMAPHORE_MAX_VALUE (15).
//   We instantiate signal_output == false, so MATH never POSTs semaphore::FPU_SFPU and there is nothing to drain on
//   that channel (matches the demo default; mirrors sdpa_custom_mm_test.cpp's "MATH never WAITS on SFPU" reasoning,
//   here inverted: MATH waits, UNPACK fakes the post). See harness_needs in the comparison verdict.
//
// Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is pending
// Blackhole hardware/CI; this host is Wormhole. The MatmulGolden below validates standard LoFi matmul numerics and,
// like the custom_mm / sdpa_custom_mm analog tests, does not model the primitive's exact partial-tile DEST layout;
// exact numerical agreement is validated only when run on Blackhole hardware.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// One output tile per K-iteration (nt_dim == 1).
static constexpr std::uint32_t NT_DIM = 1;

// DEST layout. Both src_index and dst_index are RAW DEST ROW offsets (the primitive TT_SETC16s
// DEST_TARGET_REG_CFG_MATH_Offset directly), and every tile this op touches -- in0 K-tiles and output tiles alike --
// is 16 DEST rows: an 8x32 logical tile packed into one 16x16 face, rows 0-7 holding logical columns 0-15 and rows
// 8-15 holding columns 16-31. The execute reads in0 K-tile i from src_index + i*16 and writes output tile j at
// dst_index + j*16.
//
// The in0 seed therefore needs kt_dim consecutive 16-row slots, which is exactly the 4 faces of ONE 32x32 datacopy
// tile (faces land at dest rows 0/16/32/48). So a single A2D datacopy of a 4-face tile seeds up to 4 K-tiles, which
// covers the kt_dim <= 4 grid this test sweeps. The seed goes in DEST tile 1 (rows 64+) and the output in DEST tile 0
// (rows 0+) so the pack loop can use tile indices 0..nt_dim-1 unshifted.
static constexpr std::uint32_t SRC_DEST_TILE = 1;                  // A2D datacopy tile index for the in0 seed
static constexpr std::uint32_t SRC_INDEX     = SRC_DEST_TILE * 64; // ...as a raw DEST row offset
static constexpr std::uint32_t DST_INDEX     = 0;
// kt_dim K-tiles must fit in the seed tile's 4 face slots, and the UNPACK-side semaphore fake below posts kt_dim
// times against a semaphore that saturates at SEMAPHORE_MAX_VALUE (15).
static constexpr std::uint32_t MAX_KT_DIM = 4;
// signal_output == false: full K accumulation, MATH POSTs nothing on semaphore::FPU_SFPU (no drain needed).
static constexpr bool SIGNAL_OUTPUT = false;
// output_granularity only gates the FPU_SFPU post cadence in the signal_output branch; with SIGNAL_OUTPUT == false
// it does not affect numerics. Pin to 1 (the demo-fork default). On promotion, the tt-blaze primitive additionally
// takes a defaulted input_granularity template param for the K-batched wait cadence (see promotion_notes); the
// demo-fork header templates output_granularity only.
static constexpr std::uint32_t OUTPUT_GRANULARITY = 1;

// LoFi-only demo-fork path.
static constexpr MathFidelity SDPA_FIDELITY = MathFidelity::LoFi;

#ifdef LLK_TRISC_UNPACK

// PRIMITIVE symbols under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// Resolved via the ADVANCE TEST shadow -I in test_config.py (demo-fork tt_llk root); repoint on promotion.
#include "llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h"
// Base unpack_A supplies the per-tile execute for the in0 DEST seed (the SDPA execute unpacks in1/SrcA only).
#include "llk_unpack_A.h"
// ...and llk_unpack_A_sdpa.h supplies the DEMO's SrcB-only dummy-valid helper. NOT the canonical
// _llk_unpack_set_srcb_dummy_valid_() in llk_unpack_common.h: that one is 3 instructions and sets SrcB *and* SrcA, and
// the spurious SrcA dvalid is what the seed datacopy below would consume instead of the real in0 tile. The demo
// equivalent of the canonical helper is _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_(); the one we want here is the
// SrcB-only _llk_unpack_A_sdpa_set_srcb_dummy_valid_().
#include "llk_unpack_A_sdpa.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Only SrcA (in1, the full [32,32] K-tiles) is unpacked by the SDPA execute; SrcB is reused from DEST by the MATH
    // thread. buffer_B carries in1 (SrcA), buffer_A carries the in0 tile that MATH seeds into DEST.
    //
    // Both operand-suffixed face counts are CROSSED, matching every other crossed argument in this call and the
    // silicon-validated matmul_custom_compressed_test.cpp: num_faces_B (in1, a full 4-face tile) goes to the unpA slot
    // and num_faces_A to the unpB slot. The unpA face count is the one that matters here -- it is the slot the in1
    // K-tiles and the in0 seed both unpack through.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src, // SrcA <- in1 (full K-tiles) and the in0 seed
        formats.unpack_A_src, // SrcB slot (unused by unpack here; DEST-reuse feeds SrcB)
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        params.in0_face_r_dim,
        params.num_faces_B /* unpA_num_faces (in1, full tile) */,
        params.num_faces_A /* unpB_num_faces (unused slot) */);

    // (1) dest<-SFPU handshake fake, HOISTED ABOVE every instruction that can block. MATH does
    //     t6_semaphore_wait_on_zero<STALL_MATH>(UNPACK_MATH_DONE) at the top of EVERY K-iteration, before it retires
    //     any MVMUL, so it frees no SrcA bank until the first post lands. Meanwhile the SDPA unpack execute below ends
    //     in wait_for_next_context(1), which spins on UNPACK_SYNC until its MOP retires -- and the MOP needs SrcA
    //     banks only MATH releases. Supply is 2 banks plus the one the seed datacopy releases, against 1 + kt_dim
    //     demanded: fine at kt_dim == 2, wedged at kt_dim == 4. Posting first is safe because SEMWAIT with
    //     STALL_ON_ZERO is level-triggered, so a pre-post cannot lose an edge. Interleaving one post per K-tile is not
    //     an option from the test side: the primitive collapses the whole K loop into a single MOP.
    //     MATH's matching per-K t6_semaphore_get decrements each post, so the net balance is zero.
    //     The pre-post depth also has to fit the semaphore's max: SEMPOST SATURATES silently, so if
    //     UNPACK_MATH_DONE's max is below kt_dim the surplus posts are dropped and MATH's later per-K waits never
    //     clear -- which is exactly what wedged kt_dim == 4 while kt_dim == 2 passed. Nothing in this isolated kernel
    //     SEMINITs semaphore 6 (the demo relies on its paired SFPU kernel's cadence instead), so do it here.
    LLK_ASSERT(params.KT_DIM <= MAX_KT_DIM, "kt_dim must be <= 4: the in0 seed tile has only 4 face slots");
    t6_semaphore_init(semaphore::UNPACK_MATH_DONE, 0 /* min */, semaphore::SEMAPHORE_MAX_VALUE);
    for (std::uint32_t i = 0; i < params.KT_DIM; i++)
    {
        t6_semaphore_post(semaphore::UNPACK_MATH_DONE);
    }

    // (2) Unpack the REAL in0 tile for MATH's seed datacopy. Without this the datacopy consumes whatever dvalid
    //     happens to be sitting in SrcA -- previously the spurious one from the canonical SrcB+SrcA dummy-valid
    //     helper, i.e. the ZEROSRC bank -- so in0 was identically zero and the matmul reduced to 0 regardless of
    //     stimuli, while the MatmulGolden asserted against a real product.
    _llk_unpack_A_init_<BroadcastType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_B_src, formats.unpack_B_dst);
    _llk_unpack_A_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), formats.unpack_B_src, formats.unpack_B_dst);

    // (3) EXACTLY ONE dummy SrcB SET_DVALID, using the DEMO's SrcB-ONLY helper (see the include note above).
    //     The matmul execute opens with STALLWAIT(STALL_MATH, WAIT_SFPU | SRCB_VLD) before its first MOVD2B
    //     DEST->SrcB, and the seed datacopy's MOP ends in SETRWC(CLR_AB), which retires a SrcB dvalid. One is the
    //     only working count, measured on p100a: with none, both kt_dim variants hang waiting on SRCB_VLD; with two,
    //     kt_dim == 4 hangs with all three threads stalled (SrcB has 2 banks and MATH only ever clears it twice, so
    //     the surplus wedges the unpacker against the in1 stream it still has to issue).
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();

    // (4) The in1 K-tile stream. Its init must come after the plain unpack_A init above, which reprograms the MOP.
    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(NT_DIM, FACE_R_DIM, params.num_faces_B);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        L1_ADDRESS(params.buffer_B[0]), // base_address_a -> SrcA (in1, full K-tiles)
        0 /* tile_index_a */,
        params.TILE_SIZE_UNPACK_B, // tile_size_a (in1)
        params.KT_DIM,
        NT_DIM,
        1 /* in1_k_stride: contiguous K tiles */);
}

#endif

#ifdef LLK_TRISC_MATH

// PRIMITIVE symbol under test (NOT the forked _api.h wrapper / compute_kernel_api entry).
// Resolved via the ADVANCE TEST shadow -I in test_config.py (demo-fork tt_llk root); repoint on promotion.
//
// The demo-fork header trips a pre-existing -Werror warning, inert for numerics (see comparison verdict): the
// sdpa_custom_mm_reuse_dest_srcb_configure_addrmod / _configure_mop helpers take in0/in1 tile-dim + partial_face
// params they never read (only transpose/kt_dim carry [[maybe_unused]]), tripping -Wunused-parameter under the
// harness's -Wall -Werror. These live inside template bodies reached by this TU, so suppress at file scope; an
// include-only wrap does not reach the instantiation point. Remove on promotion once the canonical header is clean.
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_sdpa_custom_mm_reuse_dest_srcb.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Seed DEST with the in0 tile that the matmul will MOVD2B into SrcB. In the real op these DEST rows are produced
    // by a prior pass; here buffer_A is unpacked into SrcA by the UNPACK thread and copied in with a plain A2D
    // datacopy so the golden is a plain matmul. The 4 faces of this one 32x32 datacopy tile land at DEST rows 0/16/32/
    // 48 of DEST tile SRC_DEST_TILE, which is exactly the 4 K-tile slots the execute reads at src_index + i*16.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(params.num_faces_B, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(SRC_DEST_TILE, formats.math, formats.math, params.num_faces_B);

    // The SDPA init MUST come after the datacopy init/execute, not before. It programs ADDR_MOD_0..3 and records the
    // MVMUL replay buffer, but _llk_math_eltwise_unary_datacopy_init_<A2D> then overwrites ADDR_MOD_0 (srca 16 -> 1,
    // dest 8 -> 1), ADDR_MOD_2 (zero -> srca+8/dest+8) and ADDR_MOD_3 (clr_ab/dest+8/c_to_cr -> all-zero), and the
    // execute below never re-inits. The replay buffer stores slot INDICES rather than contents, so every replayed
    // MVMUL and every MOVD2B in the K-reduction would run with datacopy stepping -- and math::reset_counters cannot
    // mask it, because the ADDR_MOD_0 drift happens inside a single 4-instruction replay. The three sibling tests here
    // order it this way too.
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_<SDPA_FIDELITY>(
        params.in0_face_r_dim /* in0_tile_r_dim */,
        TILE_C_DIM /* in0_tile_c_dim */,
        TILE_R_DIM /* in1_tile_r_dim */,
        TILE_C_DIM /* in1_tile_c_dim */,
        false /* partial_face */,
        0 /* transpose */,
        params.KT_DIM);

    // Custom K-reduction matmul: MOVD2B DEST->SrcB per K-tile, MVMUL(SrcA, SrcB) accumulate into DEST[DST_INDEX].
    // signal_output == false: MATH POSTs nothing on semaphore::FPU_SFPU (no drain needed); it still WAITs on
    // semaphore::UNPACK_MATH_DONE per K-tile, which the UNPACK thread fakes (see banner).
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_<OUTPUT_GRANULARITY>(SRC_INDEX, DST_INDEX, false /* transpose */, params.KT_DIM, NT_DIM, SIGNAL_OUTPUT);

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
    // Each output tile is 16 DEST rows == ONE 16x16 face (an 8x32 logical tile: rows 0-7 are logical columns 0-15,
    // rows 8-15 are columns 16-31). So the packer runs single-face, and the DEST tile-to-tile stride has to be brought
    // down from a full 32x32 tile (64 rows) to 16 rows, i.e. Wstride / 4.
    // tile_size is the single-face 16x16 output tile, NOT params.TILE_SIZE_PACK (which is still a full 32x32 tile:
    // the harness only rescales it when IN_TILE_DIMS is supplied). The host sizes buffer_Res to match via
    // StimuliConfig(operand_res_tile_size=...).
    constexpr std::uint32_t OUT_TILE_SIZE_BYTES = FACE_R_DIM * FACE_C_DIM * 2;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, OUT_TILE_SIZE_BYTES, FACE_R_DIM, FACE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, FACE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 4) * FACE_C_DIM * FACE_R_DIM * 2);
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
