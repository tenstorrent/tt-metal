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
const std::uint32_t ct_dim             = 1;
const bool UNTILIZE                    = true;
std::uint32_t face_size                = 128;
std::uint32_t tile_size                = 16 * 16 * 4;
const ckernel::DstSync sync            = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_AB_matmul_init_<>();
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_unpack_AB_matmul_<>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), 0, 0, face_size, face_size);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_matmul_init_<MATH_FIDELITY>();
    _llk_math_pack_sync_init_<sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reconfig_remap_wrapper_(true);
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<sync>();
        _llk_math_matmul_<MATH_FIDELITY>(0);
        _llk_math_dest_section_done_<sync, is_fp32_dest_acc_en>();
    }
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_dest_init_wrapper_<sync, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
    _llk_pack_untilize_init_wrapper_<ct_dim>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4 /* num_faces */);
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_untilize_wrapper_<ct_dim>(L1_ADDRESS(params.buffer_Res[block]), formats.pack_dst, FACE_R_DIM, 4 /* num_faces */, 0 /* tile_dst_rt_offset */);
        _llk_pack_dest_section_done_<sync, is_fp32_dest_acc_en>();
    }
}

#endif
