// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;
const uint32_t ct_dim             = 1;
const bool UNTILIZE               = true;
uint32_t face_size                = 128;
uint32_t tile_size                = 16 * 16 * 4;
const ckernel::DstSync sync       = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "params.h"

void run_kernel()
{
    _llk_unpack_AB_matmul_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_B_IN, UNPACK_A_OUT, UNPACK_B_OUT);
    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]), 0, 0, face_size, face_size);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel()
{
    _llk_math_matmul_init_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>();
    _llk_math_pack_sync_init_<sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<true, false>(MATH_FORMAT, MATH_FORMAT);
    _llk_math_wait_for_dest_available_<sync>();
    _llk_math_matmul_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>(0);
    _llk_math_dest_section_done_<sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(PACK_IN, PACK_OUT, tile_size);
    _llk_pack_dest_init_<sync, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
    _llk_pack_untilize_init_<ct_dim>(PACK_IN, PACK_OUT, FACE_R_DIM, 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(PACK_IN, PACK_OUT, tile_size);
    _llk_pack_dest_init_<sync, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIZE>();
    _llk_pack_untilize_init_<ct_dim>(PACK_OUT, FACE_R_DIM, 4);
#endif
    _llk_packer_wait_for_math_done_();
    _llk_pack_untilize_<ct_dim>(L1_ADDRESS(buffer_Res[0]), PACK_OUT, FACE_R_DIM, 4, 0);
    _llk_pack_dest_section_done_<sync, is_fp32_dest_acc_en>();
}

#endif
