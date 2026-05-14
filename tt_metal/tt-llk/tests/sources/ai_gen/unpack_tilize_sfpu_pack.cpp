// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Configure unpack for tilize operation (row-major input -> tiled format)
    // This handles both A and B inputs which need to be tilized before binary ops
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */);

    // Unpack and tilize single tile A (stored in src A register - index 0)
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_A[0]),
        0 /* tile_index */,
        formats.unpack_A_src,
        formats.unpack_A_dst,
        BLOCK_CT_DIM,
        FACE_R_DIM,
        4 /* num_faces */,
        false /* narrow_tile */);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    // Initialize datacopy operation (copy src A to dest)
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* tilize */, false /* is_int_fpu_en */>(
        4 /* num_faces */, formats.math);

    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // Wait for destination to be available
    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Step 1: Copy tilized input from src A to dest
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, formats.math, formats.math);

    // Step 2: Initialize and perform SFPU unary operation on the copied data
    test_utils::call_unary_sfpu_operation_init<SFPU_UNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en, 32>();
    test_utils::call_unary_sfpu_operation<DST_SYNC, is_fp32_dest_acc_en, SFPU_UNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en, 32>(0, formats.math);

    // Signal completion to packer
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Configure packer hardware for standard pack (no untilize)
    const bool UNTILIZE = false;
    const bool TILIZE   = false; // Input to pack is already in tile format

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, UNTILIZE, TILIZE>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<UNTILIZE, false /* zero_output */, TILIZE>(formats.pack_dst);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Pack the single result tile from destination register to output buffer
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, UNTILIZE>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
