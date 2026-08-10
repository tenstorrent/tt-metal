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

// Single output tile [1,32]: ct_dim==1, rt_dim==1, one SrcA tile per K-iteration.
static constexpr std::uint32_t NT_DIM            = 1;
static constexpr std::uint32_t SRC_INDEX         = 0;
static constexpr std::uint32_t DST_INDEX         = 0;
// signal_output == false: full K accumulation, MATH POSTs nothing on semaphore::FPU_SFPU (no drain needed).
static constexpr bool SIGNAL_OUTPUT              = false;
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
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Only SrcA (in1, the full [32,32] K-tiles) is unpacked; SrcB is reused from DEST by the MATH thread. in0 (the
    // [1,32] partial tile) is seeded into DEST on the MATH thread. buffer_B carries in1 (SrcA), buffer_A carries in0.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,  // SrcA <- in1 (full K-tiles)
        formats.unpack_A_src,  // SrcB slot (unused by unpack here; DEST-reuse feeds SrcB)
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        params.IN0_FACE_R_DIM,
        params.num_faces_A /* unpA_num_faces (in1, full tile) */,
        params.num_faces_B);

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(NT_DIM, FACE_R_DIM, params.num_faces_A);

    // Inject the dummy SrcB SET_DVALID the MATH preamble's STALLWAIT(SRCB_VLD) waits on before its MOVD2B DEST->SrcB.
    // (The demo compute API calls llk_unpack_A_sdpa_set_srcb_dummy_valid() here; the canonical primitive in
    // llk_unpack_common.h is byte-equivalent — same STALL_UNPACK + SrcB/SrcA SET_DVALID UNPACR_NOPs.)
    _llk_unpack_set_srcb_dummy_valid_();

    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        L1_ADDRESS(params.buffer_B[0]),  // base_address_a -> SrcA (in1, full K-tiles)
        0 /* tile_index_a */,
        params.TILE_SIZE_UNPACK_B,  // tile_size_a (in1)
        params.KT_DIM,
        NT_DIM,
        1 /* in1_k_stride: contiguous K tiles */);

    // dest<-SFPU handshake fake: post semaphore::UNPACK_MATH_DONE once per K-tile so MATH's per-K
    // t6_semaphore_wait_on_zero(UNPACK_MATH_DONE) clears (see banner). MATH's matching per-K get decrements each post,
    // so the net balance is zero. Without this the isolated MATH thread blocks forever.
    for (std::uint32_t i = 0; i < params.KT_DIM; i++)
    {
        t6_semaphore_post(semaphore::UNPACK_MATH_DONE);
    }
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

    _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_<SDPA_FIDELITY>(
        params.IN0_FACE_R_DIM /* in0_tile_r_dim */,
        TILE_C_DIM /* in0_tile_c_dim */,
        TILE_R_DIM /* in1_tile_r_dim */,
        TILE_C_DIM /* in1_tile_c_dim */,
        false /* partial_face */,
        0 /* transpose */,
        params.KT_DIM);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Seed DEST with the in0 [1,32] source tile (unpacked into SrcA above via the base execute is not available here;
    // instead we rely on the reuse-from-DEST contract). In the real op the DEST source rows are produced by a prior
    // pass; here we feed buffer_A's tile directly with an A2D datacopy so the golden is a plain matmul. The MATH
    // primitive reads DEST rows into SrcB via MOVD2B keyed off src_index + i*16 per K-iteration.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(
        params.num_faces_B, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en>(
        SRC_INDEX, formats.math, formats.math, params.num_faces_B);

    // Custom K-reduction matmul: MOVD2B DEST->SrcB per K-tile, MVMUL(SrcA, SrcB) accumulate into DEST[DST_INDEX].
    // signal_output == false: MATH POSTs nothing on semaphore::FPU_SFPU (no drain needed); it still WAITs on
    // semaphore::UNPACK_MATH_DONE per K-tile, which the UNPACK thread fakes (see banner).
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_<OUTPUT_GRANULARITY>(
        SRC_INDEX, DST_INDEX, false /* transpose */, params.KT_DIM, NT_DIM, SIGNAL_OUTPUT);

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
