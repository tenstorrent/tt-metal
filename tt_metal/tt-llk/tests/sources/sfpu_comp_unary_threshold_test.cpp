// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Threshold decoding of the tt-llk scalar compares (tt-llk#1701 item 16).
//
// Drives the uint32 entry point _calculate_comp_unary_<APPROX, unary_*>(value) over one tile. Like the other
// float-scalar kernels (metal calculate_unary_*, tt-llk relu/threshold), `value` carries the fp32 bit pattern
// of the threshold (SFPU_UNARY_SCALAR), e.g. 0x3F000000 for 0.5f. With SFPU_UNARY_SCALAR_AT_RUNTIME the
// threshold is read from the runtime parameters instead, so nothing the kernel decides from the threshold's
// value can fold at compile time: the same ELF serves every threshold, as it does for a caller that passes a
// runtime scalar.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// The tile the datacopy writes and the SFPU then reads; both calls must use the same index.
static constexpr std::uint32_t DST_INDEX = 0;

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

    _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_comp.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // The compares need only the generic SFPU init (config reg, ADDR_MOD_7, dest RWC reset).
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    _llk_math_wait_for_dest_available_<dest_sync>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        DST_INDEX, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

#if defined(SFPU_UNARY_SCALAR_AT_RUNTIME) && !defined(SPEED_OF_LIGHT)
    const std::uint32_t threshold = params.SFPU_UNARY_SCALAR;
#else
    constexpr std::uint32_t threshold = SFPU_UNARY_SCALAR;
#endif

    // ITERATIONS=8 per face at VectorMode::RC covers the whole tile.
    _llk_math_eltwise_unary_sfpu_params_(
        [](std::uint32_t value) { ckernel::sfpu::_calculate_comp_unary_<APPROX_MODE, SFPU_UNARY_OPERATION, 8 /* ITERATIONS */>(value); },
        DST_INDEX,
        VECTOR_MODE,
        threshold);

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
