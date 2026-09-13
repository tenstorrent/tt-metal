// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression: sinkhorn must leave the shared -1.0 constant register intact.
//
// _sinkhorn_program_parity_mask_ writes its lane-parity mask into programmable constant
// vector register 11 with TTI_SFPCONFIG. p_sfpu::LCONST_neg1 is that same register, and
// it is a core-wide constant every other SFPU kernel reads as -1.0. The file's own
// comment states the constraint ("this clobbers the HW -1.0 constant") but scopes it to
// sinkhorn's own row-norm body; nothing takes the mask down when sinkhorn returns, and
// sinkhorn is the only site in the tree that writes register 11 at all.
//
// This driver composes two ops the public compute API exposes side by side, in the order
// a kernel would call them, and checks only the SECOND one:
//
//   dest 0: sinkhorn_4x4  -- runs for its side effect on register 11; its own numeric
//                            output is not read by this test
//   dest 1: floor         -- ckernel_sfpu_rounding_ops.h uses LCONST_neg1 to apply the
//                            "if v > trunc(v), subtract one" correction
//
//   register 11 intact:   floor(x)          e.g. floor(-2.5) == -3.0
//   register 11 clobbered: the correction adds the parity mask (integer 0 / 2, which as
//                          a float is zero or a denormal) instead of -1.0, so floor
//                          degenerates to trunc:            floor(-2.5) == -2.0
//
// The two outcomes differ by exactly 1.0 on every negative non-integer, which is far
// outside any tolerance, so the python side asserts exact equality.
//
// Blackhole only: sinkhorn lives in the Blackhole experimental SFPU tree.

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
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    // The same tile twice: dest 0 feeds sinkhorn, dest 1 feeds floor.
    for (std::uint32_t i = 0; i < 2; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"
#include "sfpu/experimental/ckernel_sfpu_sinkhorn.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(1, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // Run sinkhorn exactly as the compute API dispatches it. One iteration and one face
    // are enough: this test is about the constant register it leaves behind, not its
    // numerics, and a short run keeps the driver fast.
    // Same dispatch llk_math_sinkhorn.h performs for sinkhorn_4x4(); the wrapper header
    // itself is not included because its unused init helper does not resolve here.
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::
            _sinkhorn_4x4_<1 /* NUM_FACES_USED */, 1 /* ITERS */, 0x35890000 /* EPS_BITS */, false /* SINGLE_SUBMAT */, 32 /* VALID_H */, 32 /* VALID_W */>,
        0 /* input_index */,
        VectorMode::RC_custom);

    // floor on the untouched second tile. ITERATIONS=8 at VectorMode::RC is what
    // floor_tile dispatches.
    _llk_math_eltwise_unary_sfpu_params_(
        [] { ckernel::sfpu::_calculate_floor_<false /* APPROXIMATION_MODE */, 8 /* ITERATIONS */>(); }, 1 /* dst_index */, VectorMode::RC);

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
    // Only the floor result is validated.
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
