// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Zero-base coverage for _sfpu_binary_power_.
//
// Driven through calculate_rpow rather than calculate_sfpu_binary_pow because rpow takes its
// base as a scalar and its exponent from Dest. SFPU_UNARY_SCALAR pins the base at +-0.0 and
// the input tile carries the exponent classes. Both entry points call _sfpu_binary_power_<is_fp32_dest_acc_en>.
//
// Note: MathOperation.SfpuElwpow routes to BinaryOp::POW which is a separate implementation in ckernel_sfpu_binary.h
// Also, sfpu_operations.h hardcodes rpow's base to 2.0f.

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

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

#include "llk_sfpu/ckernel_sfpu_rpow.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // The init sfpu_operations.h routes rpow to: it primes vConstFloatPrgm0/1/2, which the
    // log2/exp2 pair in _sfpu_binary_power_ reads.
    llk_math_eltwise_unary_sfpu_init<SfpuType::rpow>(sfpu::sfpu_binary_pow_init<APPROX_MODE>);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // The exponent tile lands in Dest tile 0; calculate_rpow then overwrites it in place with
    // base**exponent for the compile-time base.
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);

    // Rebase the Dest RWC before the sfpi-based op
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // ITERATIONS=8 at VectorMode::RC is what rpow_tile dispatches.
    SFPU_UNARY_CALL(
        DST_SYNC,
        is_fp32_dest_acc_en,
        calculate_rpow,
        (APPROX_MODE, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        0 /* dst_index */,
        VECTOR_MODE,
        SFPU_UNARY_SCALAR);

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
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
