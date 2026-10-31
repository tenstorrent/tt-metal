// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// calculate_unary_power_iterative over one tile, with the exponent either read from the runtime
// parameter block (POWER_EXPONENT, opaque to the compiler: the rolled, runtime-exponent loop that moreh's
// power_tile_to_cb reaches) or emitted as a compile-time constant (POWER_CONST_EXPONENT: the unrolled loop
// that ttnn's power_iterative_tile(i, Nu) reaches). POWER_EXPONENT_IS_CONSTANT picks the source. Init and
// call mirror power_iterative_tile_init() / power_iterative_tile().

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

// _llk_unpack_A_init_ and _llk_unpack_A_ must be given the same template arguments.
static constexpr bool ACC_TO_DEST = false;

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

    _llk_unpack_A_init_<BroadcastType::NONE, ACC_TO_DEST, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, ACC_TO_DEST, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

// After the macros header, which forward-declares the per-op inits; this one defines power_init.
#include "llk_sfpu/ckernel_sfpu_unary_power.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    // --speed-of-light emits POWER_EXPONENT as a constexpr and leaves RuntimeParams empty.
    const std::uint32_t POWER_EXPONENT = params.POWER_EXPONENT;
#endif
    const std::uint32_t exponent = POWER_EXPONENT_IS_CONSTANT ? POWER_CONST_EXPONENT : POWER_EXPONENT;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_wait_for_dest_available_<dest_sync>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        DST_INDEX, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // power_iterative_tile_init(): the datacopy above programs the FPU addr_mods, so the SFPU init runs after it.
    ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::power, is_fp32_dest_acc_en>();

    // power_iterative_tile(): ITERATIONS=8 per face at VectorMode::RC covers the whole tile.
    SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_unary_power_iterative, (APPROX_MODE, 8 /* ITERATIONS */), DST_INDEX, VectorMode::RC, exponent);

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
