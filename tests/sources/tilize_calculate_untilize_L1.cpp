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

#ifdef DEST_ACC
const bool is_fp32_dest_acc_en = true;
#else
const bool is_fp32_dest_acc_en = false;
#endif

volatile uint32_t* const buffer_A = reinterpret_cast<volatile uint32_t*>(0x1a000);
volatile uint32_t* const buffer_B = reinterpret_cast<volatile uint32_t*>(0x1b000);

volatile uint32_t* const buffer_A_tilized = reinterpret_cast<volatile uint32_t*>(0x1c000);
volatile uint32_t* const buffer_B_tilized = reinterpret_cast<volatile uint32_t*>(0x1d000);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel()
{
    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_A_OUT, FACE_R_DIM, 0, 4);

    _llk_unpack_tilize_init_(UNPACK_A_IN, UNPACK_A_OUT, 1, FACE_R_DIM, false);
    _llk_unpack_tilize_(L1_ADDRESS(buffer_A), 0, UNPACK_A_IN, 1, FACE_R_DIM, 4, false);

    _llk_unpack_tilize_init_(UNPACK_B_IN, UNPACK_B_OUT, 1, FACE_R_DIM, false);
    _llk_unpack_tilize_(L1_ADDRESS(buffer_B), 0, UNPACK_B_IN, 1, FACE_R_DIM, 4, false);

    _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_B_IN, UNPACK_A_OUT, UNPACK_B_OUT, FACE_R_DIM, 0, 4);
    _llk_unpack_AB_init_<>();
    _llk_unpack_AB_<>(L1_ADDRESS(buffer_A_tilized), L1_ADDRESS(buffer_B_tilized));
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
    const bool is_int_fpu_en                = false;
    const std::uint32_t operand_A_dst_index = 1;
    const std::uint32_t operand_B_dst_index = 2;
    const std::uint32_t res_dst_index       = 0;
    const bool TILIZE                       = true;

// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE, TILIZE, is_fp32_dest_acc_en, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE, is_fp32_dest_acc_en, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#endif

    _llk_math_pack_sync_init_<DstSync::SyncFull, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FORMAT, MATH_FORMAT);

    // copy tilized inputs to dest indexes 0 and 1
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, is_fp32_dest_acc_en, false>(
        operand_A_dst_index, MATH_FORMAT, MATH_FORMAT);
    _llk_math_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();

    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, is_fp32_dest_acc_en, false>(
        operand_B_dst_index, MATH_FORMAT, MATH_FORMAT);
    _llk_math_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();

    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BroadcastType::NONE, MATH_FIDELITY>(4, 0, 0);
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BroadcastType::NONE, DstSync::SyncFull, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE, is_fp32_dest_acc_en>(
        4, res_dst_index, false);
    _llk_math_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    volatile uint32_t* const buffer_Dest    = reinterpret_cast<volatile uint32_t*>(0x1e000);
    const std::uint32_t ct_dim              = 1;
    const std::uint32_t operand_A_dst_index = 1;
    const std::uint32_t operand_B_dst_index = 2;
    const std::uint32_t res_dst_index       = 0;
    const bool UNTILIZE                     = false;
    const bool TILIZE                       = true;

    std::fill(buffer_Dest, buffer_Dest + 16 * 16 * 4, 0xdeadbeef);

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<UNTILIZE, is_fp32_dest_acc_en, TILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false, TILIZE>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<UNTILIZE, is_fp32_dest_acc_en>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, UNTILIZE, is_fp32_dest_acc_en>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull, UNTILIZE, is_fp32_dest_acc_en>(operand_A_dst_index, L1_ADDRESS(buffer_A_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull, UNTILIZE, is_fp32_dest_acc_en>(operand_B_dst_index, L1_ADDRESS(buffer_B_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();

    // Needed to reconfigure pack for regular not tilized pack for BH

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<UNTILIZE, is_fp32_dest_acc_en, !TILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIZE, false, DstTileFaceLayout::RowMajor, false, !TILIZE>(PACK_OUT);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull, UNTILIZE, is_fp32_dest_acc_en>(res_dst_index, L1_ADDRESS(buffer_Dest));
    _llk_pack_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();
}

#endif
