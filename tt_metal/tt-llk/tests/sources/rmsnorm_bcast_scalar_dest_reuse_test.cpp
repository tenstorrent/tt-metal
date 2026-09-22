// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the rmsnorm bcast-scalar dest-reuse family promoted by tt-metal #52709
// (api/compute/experimental/rmsnorm.h -> _llk_math_rmsnorm_bcast_scalar_dest_reuse_ /
// _llk_unpack_A_rmsnorm_init_).
//
// Why this needs its own driver rather than an extension of an existing binary test: the
// op is a num_tiles-templated MOP driven from a SINGLE unpack call, with SrcB sourced from
// DEST via MOVD2B under a WAIT_SFPU|SRCB_VLD stall rather than from L1. test_bcast.py does
// one-tile-per-unpack broadcasts and test_eltwise_binary.py has neither the
// num_tiles-as-template-argument plumbing nor a MOP-over-N-tiles axis.
//
// Sequence, mirroring rmsnorm_bcast_scalar_reuse_tiles* in the compute API:
//
//   seed   A plain A2D datacopy lands one tile in DEST[0]. Element [0] of that tile is the
//          value MOVD2B later broadcasts -- in the real kernel this slot holds 1/RMS,
//          produced by add_rsqrt. The rest of the tile is deliberately non-uniform so a
//          MOVD2B that grabbed the wrong row/face shows up as a mismatch rather than
//          silently agreeing.
//
//   op     _llk_math_rmsnorm_bcast_scalar_dest_reuse_ with src_index == dst_index == 0,
//          exactly as unified_kernels/rmsnorm.hpp:146 calls it. MOVD2B pulls the scalar out
//          of DEST[0] into SrcB, the optional ZEROACC clears DEST, then the MOP walks
//          RMSNORM_NUM_TILES tiles of SrcA (unpacked from L1 in one _llk_unpack_A_ call)
//          against that one broadcast scalar.
//
//   pack   RMSNORM_NUM_TILES tiles.
//
// Note on CLEAR_DEST: the MOP runs with acc_to_dest=0, so it OVERWRITES the DEST rows it
// covers and the flag cannot change those. What it does change is the rows the MOP does not
// cover -- with RMSNORM_NUM_FACES < 4 the tail faces keep the seeded datacopy unless
// ZEROACC wiped them. That is the only configuration in which this axis is observable, and
// it is why the python side asserts a different golden tail per polarity.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

// The experimental rmsnorm LLK headers carry unused locals and parameters that predate this
// test: nothing outside the Metal JIT build compiled them, and that build does not treat
// -Wunused-* as errors. The tt-llk harness does (-Wall -Werror -Wunused-parameter), so the
// two diagnostics are suppressed across this include alone -- neither editing the kernel nor
// relaxing the flag for every test.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_rmsnorm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    // ---- seed: one ordinary tile, which the math thread datacopies into DEST[0] ----
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);

    // ---- the op under test ----
    // acc_to_dest=true and DEST_TO_SRCB are forced by the static_assert in
    // _llk_unpack_A_rmsnorm_mop_config_; SCALAR is what routes the base address to upk0
    // (SrcA) while SrcB is fed a dummy dvalid for the MOVD2B on the math thread.
    _llk_unpack_A_rmsnorm_init_<RMSNORM_NUM_TILES, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        RMSNORM_UNPACK_FULL_TRANSPOSE /* transpose_of_faces */, RMSNORM_UNPACK_FULL_TRANSPOSE /* within_face_16x16_transpose */, FACE_R_DIM, RMSNORM_NUM_FACES);

    // ONE call: the MOP itself walks all RMSNORM_NUM_TILES tiles from this base address.
    _llk_unpack_A_<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

// The experimental rmsnorm LLK headers carry unused locals and parameters that predate this
// test: nothing outside the Metal JIT build compiled them, and that build does not treat
// -Wunused-* as errors. The tt-llk harness does (-Wall -Werror -Wunused-parameter), so the
// two diagnostics are suppressed across this include alone -- neither editing the kernel nor
// relaxing the flag for every test.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Seed DEST[0]; element [0] of it becomes the broadcast scalar.
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);

    _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<ELTWISE_BINARY_OP, RMSNORM_NUM_TILES, MATH_FIDELITY>(RMSNORM_NUM_FACES, 0 /* acc_to_dest */);

    _llk_math_rmsnorm_bcast_scalar_dest_reuse_<ELTWISE_BINARY_OP, RMSNORM_NUM_TILES, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, RMSNORM_CLEAR_DEST>(
        0 /* src_index */, 0 /* dst_index */);

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
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t tile = 0; tile < RMSNORM_NUM_TILES; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }

    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
