// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole-only compile + golden test for the experimental LLK
//   sdpa_custom_mm_reuse_dest_srcb
// (llk_lib/experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb.h and
//  llk_lib/experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h,
//  compute API api/compute/experimental/sdpa_custom_mm_reuse_dest_srcb.h,
//  promoted by tt-metal #53295).
//
// -----------------------------------------------------------------------------
// What the op computes (the "OV" matmul of flash-attention SDPA)
// -----------------------------------------------------------------------------
// This is the second matmul of an SDPA chunk: O = P * V, where
//   P (the softmax probabilities) is ALREADY sitting in DEST -- it is the output
//     of the earlier QK^T -> reduce-max -> exp pipeline. The reuse_dest_srcb op
//     therefore does NOT unpack B from L1; instead the math thread pulls the B
//     operand out of DEST into SrcB with MOVD2B (see the math header).
//   V is unpacked from L1 into SrcA by the reuse unpack LLK.
//   O accumulates into DEST at dst_index.
//
// Geometry read straight off the math header
// (_llk_math_sdpa_custom_mm_reuse_dest_srcb_):
//   * src_index  = DEST base of the P operand (mm1_dst_offset in sdpa.h)
//   * dst_index  = DEST base of the O accumulator (mm2_dst_offset in sdpa.h)
//   * for each K iteration i in [0, kt_dim):
//       - DEST target reg is set to (src_index + i*8*2) = src_index + i*16 rows,
//         so K-chunk i of P lives in the i-th 16-row DEST region;
//       - four MOVD2B(MOV_4_ROWS) calls copy those 16 rows of P from DEST into
//         SrcB (SRC_ZERO_OFFSET + {0,4,8,12});
//       - DEST target reg is set back to dst_index;
//       - the 4-MVMUL replay buffer is issued nt_dim times, each producing /
//         accumulating one 32-wide output tile (addr_mods ADDR_MOD_0/1/3 walk
//         SrcA by 16, SrcB by 8, DEST by 8, returning to tile base between the
//         face-pairs, and advancing to the next output tile with c_to_cr).
//   * signal_output=false (used here) skips the FPU->SFPU semaphore posting, so
//     no downstream SFPU consumer is required and the op is self-contained.
//   * output_granularity/input_granularity only gate the semaphore handshake
//     cadence; with signal_output=false they do not change the numeric result.
//     We instantiate the header default (output_granularity=1, input_granularity=1).
//
// Because P is reused FROM DEST, the driver must first place P there. The math
// thread does a plain A2D datacopy of the B stimulus (the "P" matrix) into DEST
// at src_index BEFORE calling the reuse matmul, mirroring how a real SDPA chunk
// leaves exp'd scores in DEST for the OV matmul to consume.
//
// -----------------------------------------------------------------------------
// Golden (python side)
// -----------------------------------------------------------------------------
// The header comment states the op is a tiled matmul (P * V) with the single
// output tile shape [1, 32]. The golden is therefore the standard MatmulGolden
// tiled A*B (A = P from DEST, B = V from L1), tilized, with the same fidelity
// masking as every other FPU matmul in this suite. Only the DEFINED output rows
// (the first face-row band the op writes for a [1,32] tile) are validated -- the
// remaining DEST rows are left undefined by the op and must not be asserted on.
//
// NO Blackhole card is available in this environment, so this test is validated
// only at (a) a clean BH compile and (b) golden-mirrors-header inspection. The
// exact DEST<->SrcB row walk should be confirmed on a BH p100a before this test's
// numeric assertion is trusted; see the python xfail/notes.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// DEST addressing: the reuse matmul writes src_index/dst_index straight into the
// math DEST_TARGET_REG "Offset" field (TT_SETC16 DEST_TARGET_REG_CFG_MATH_Offset).
// That field is NOT a 32-row tile index: it counts DEST in half-tile (64-datum)
// units, so physical_tile = index / 64. The K-chunk stride src_index + i*8*2 puts
// chunk i at 16-datum granularity (chunk 0 at datum 0, chunk 1 at datum 16), i.e.
// both KT_DIM=2 chunks live inside physical tile 0 -- exactly one datacopy tile.
// The pack/datacopy LLKs, by contrast, address DEST in whole 32x32 TILE units.
// SRC_INDEX/DST_INDEX are the DEST_TARGET_REG offsets the matmul consumes;
// SRC_TILE/DST_TILE are the matching datacopy/pack tile bases (= offset / 64).
//
// The DST_TILE=2 (from DST_INDEX=128 -> tile 2) mapping was pinned on ttsim by
// packing all of DEST and finding the O accumulator at physical tile 2 (a
// tile-0 preload of P plus a tile-2 O accumulator, non-overlapping). Confirm the
// 64-datum offset unit on a BH p100a before trusting the exact placement.
constexpr std::uint32_t SRC_INDEX = 0;      // P DEST_TARGET offset (SrcB source) -> tile 0
constexpr std::uint32_t SRC_TILE  = 0;      // P tile base for datacopy preload
constexpr std::uint32_t DST_INDEX = 64 * 2; // O DEST_TARGET offset -> physical tile 2
constexpr std::uint32_t DST_TILE  = 2;      // O tile base for pack (= DST_INDEX / 64)

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h"
#include "experimental/llk_unpack_A_sdpa.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // hw_configure: the compute API intentionally swaps in0/in1 (see
    // sdpa_custom_mm_reuse_dest_srcb_block_init), but for the reuse op only SrcA
    // (V) is ever unpacked from L1, so configure SrcA with the V (buffer_A) format.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* unpA_num_faces */,
        4 /* unpB_num_faces */,
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B);

    // -- P (=B) preload into DEST: the reuse matmul reads SrcB from DEST, so the
    // math thread first datacopies P into DEST. That datacopy needs its own SrcA
    // stream, unpacked here with the plain unpack_A LLK. One datacopy tile holds
    // two 16-row P K-chunks, so ceil(KT_DIM/2) P tiles cover all KT_DIM chunks.
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t k = 0; k < (KT_DIM + 1) / 2; ++k)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            L1_ADDRESS(params.buffer_B[k]), formats.unpack_A_src, formats.unpack_A_dst);
    }

    // -- reuse matmul: unpack V (=SrcA, from buffer_A) via the reuse unpack LLK.
    // The unpack sets a dummy SrcB valid (SrcB comes from DEST, not L1), then the
    // single collapsed MOP walks all KT_DIM x NT_DIM SrcA tiles.
    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(NT_DIM, FACE_R_DIM, 4 /* unpA_num_faces */);
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();
    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        L1_ADDRESS(params.buffer_A[0]), 0 /* tile_index_a */, params.TILE_SIZE_UNPACK_A, KT_DIM, NT_DIM, 1 /* in1_k_stride: contiguous V tiles */);
}

#endif

#ifdef LLK_TRISC_MATH

// The promoted header's sdpa_custom_mm_reuse_dest_srcb_configure_mop() declares
// in0/in1 tile dims and partial_face but never uses them, and (unlike its siblings)
// does not tag them [[maybe_unused]]. Under the suite's -Werror=unused-parameter the
// template body fails to compile when instantiated. GCC binds the unused-parameter
// diagnostic to the parameter's DECLARATION site, so the suppression must wrap the
// header include (a call-site pragma has no effect). Drop this once the header tags
// those parameters [[maybe_unused]] -- see paramClassWrong note.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
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

    // -- P (=B) preload: datacopy the P stimulus into DEST at tile base SRC_TILE.
    // dst_index here is a TILE index (datacopy writes 32x32 tiles), so ceil(KT_DIM/2)
    // tiles hold all KT_DIM 16-row P chunks the matmul reads at (SRC_INDEX + i*16).
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    for (std::uint32_t k = 0; k < (KT_DIM + 1) / 2; ++k)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            SRC_TILE + k, formats.math, formats.math);
    }

    // -- reuse matmul: SrcB pulled from DEST at SRC_INDEX, SrcA (=V) from the
    // reuse unpack, accumulating O into DEST at DST_INDEX. signal_output=false ->
    // no FPU->SFPU handshake, so the op runs standalone. output_granularity=1,
    // input_granularity=1 are the header defaults.
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_<MATH_FIDELITY>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, KT_DIM);
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_<1 /* output_granularity */, 1 /* input_granularity */>(
        SRC_INDEX, DST_INDEX, false /* transpose */, KT_DIM, NT_DIM, false /* signal_output */);

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

    // Producer half of the SFPU->FPU input handshake. The math header consumes
    // UNPACK_MATH_DONE (aliased SFPU_FPU) once per K iteration -- a wait_on_zero +
    // t6_semaphore_get pair gated only on input_granularity, NOT on signal_output.
    // In a real SDPA chunk the SFPU exp producer posts this token to tell the FPU
    // "P is ready in DEST for K-chunk i"; the reuse matmul is the CONSUMER half of a
    // cross-layer handshake (see the semaphore-handshake-audit skill and the
    // llk_unpack_AB_reduce_custom_runtime consumer). With no SFPU partner in this
    // standalone test the math thread would wait_on_zero forever, so this spectator
    // pack thread stands in for the SFPU producer and posts KT_DIM tokens up front --
    // exactly the reduce_block_max_test.cpp pattern. Posting before wait_for_math_done
    // (and before the math waits drain them) keeps the value >=1 until each get lands.
    t6_semaphore_init(ckernel::semaphore::UNPACK_MATH_DONE, 0, KT_DIM);
    for (std::uint32_t i = 0; i < KT_DIM; ++i)
    {
        t6_semaphore_post<>(ckernel::semaphore::UNPACK_MATH_DONE);
    }

    _llk_packer_wait_for_math_done_();

    // Pack the NT_DIM output tiles starting at tile base DST_TILE.
    for (std::uint32_t i = 0; i < NT_DIM; ++i)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_TILE + i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
