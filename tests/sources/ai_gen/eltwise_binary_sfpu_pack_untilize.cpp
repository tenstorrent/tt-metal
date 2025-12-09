// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

// Constants for packer configuration
const uint32_t ct_dim    = 1; // Only one column tile (32×32 tensor)
const bool UNTILIZE      = true;
const uint32_t tile_size = 16 * 16 * 4; // bytes per face

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    // Configure unpacker for two-input AB operation (single tile each)
    _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst);
    _llk_unpack_AB_init_<>();

    // Unpack one tile from each input buffer (A and B)
    _llk_unpack_AB_<>(L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]));
}

#endif // LLK_TRISC_UNPACK

// Replace the MATH section to perform SFPU binary op on the two operands copied to dest
#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpu_operations.h"

void run_kernel()
{
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(formats.math, formats.math);

    // Binary element-wise (FPU)
    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE, MATH_FIDELITY>(4, 0);

    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        4, 0, false);

    // SFPU unary on result in dest
    _llk_math_eltwise_unary_sfpu_init_<SFPU_UNARY_OPERATION>();
    _llk_math_eltwise_unary_sfpu_start_<DST_SYNC>(0);

    test_utils::call_sfpu_operation_32(SFPU_UNARY_OPERATION);

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
    _llk_pack_untilize_init_<ct_dim>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIZE>();
    _llk_pack_untilize_init_<ct_dim>(formats.pack_dst, FACE_R_DIM, 4);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_untilize_<ct_dim>(L1_ADDRESS(buffer_Res[0]), formats.pack_dst, FACE_R_DIM, 4, 0);
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
