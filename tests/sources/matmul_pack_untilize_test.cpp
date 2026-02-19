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

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(params->buffer_A[0]), L1_ADDRESS(params->buffer_B[0]), 0, 0, face_size, face_size);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_math_matmul_init_<MATH_FIDELITY>();
    _llk_math_pack_sync_init_<sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
#ifdef ARCH_BLACKHOLE
    _llk_math_reconfig_remap_(true);
#endif
    _llk_math_wait_for_dest_available_<sync>();
    _llk_math_matmul_<MATH_FIDELITY>(0);
    _llk_math_dest_section_done_<sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_dest_init_<sync, is_fp32_dest_acc_en>();
    _llk_pack_untilize_init_<ct_dim>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_dest_init_<sync, is_fp32_dest_acc_en, UNTILIZE>();
    _llk_pack_untilize_init_<ct_dim>(formats.pack_dst, FACE_R_DIM, 4);
#endif
    _llk_packer_wait_for_math_done_();
    _llk_pack_untilize_<ct_dim>(L1_ADDRESS(params->buffer_Res[0]), formats.pack_dst, FACE_R_DIM, 4, 0);
    _llk_pack_dest_section_done_<sync, is_fp32_dest_acc_en>();
}

#endif
