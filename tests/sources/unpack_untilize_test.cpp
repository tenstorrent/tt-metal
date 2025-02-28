// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <algorithm>

#include "llk_defs.h"
#include "ckernel.h"

// Globals
uint32_t unp_cfg_context = 0;
uint32_t pack_sync_tile_dst_ptr = 0;
volatile uint32_t tt_l1_ptr l1_buffer[16] __attribute__ ((section (".text#"))) __attribute__ ((aligned (16)));

#ifdef DEST_ACC
const bool is_fp32_dest_acc_en = true;
#else
const bool is_fp32_dest_acc_en = false;
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_untilize.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    volatile uint32_t* const buffer_A = reinterpret_cast<volatile uint32_t*>(0x1a000);

    _llk_unpack_untilize_hw_configure_<is_fp32_dest_acc_en,StochRndType::None>(DATA_FORMAT, DATA_FORMAT, FACE_R_DIM, 0, 4);
    _llk_unpack_untilize_init_(DATA_FORMAT, 1024, FACE_R_DIM, 4);
    _llk_unpack_untilize_pass_<true>(L1_ADDRESS(buffer_A),1);
    _llk_unpack_untilize_pass_<false>(L1_ADDRESS(buffer_A),1);
}

#endif


#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
    // copy srca to dest
    #ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE,false, is_fp32_dest_acc_en, is_int_fpu_en>(0, 0, 4, DATA_FORMAT);
    #else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE, is_fp32_dest_acc_en, is_int_fpu_en>(0, 0, 4, DATA_FORMAT);
    #endif
    _llk_math_pack_sync_init_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT, DATA_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, is_fp32_dest_acc_en, false>(0, DATA_FORMAT, DATA_FORMAT);
    _llk_math_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
}

#endif 

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    volatile uint32_t* const buffer_Dest = reinterpret_cast<volatile uint32_t*>(0x1c000);
    const bool UNTILIIZE = false;

    std::fill(buffer_Dest, buffer_Dest + 16 * 16 * 4, 0xdeadbeef);
    
    #ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<UNTILIIZE, is_fp32_dest_acc_en, false>(DATA_FORMAT, DATA_FORMAT, 16*16*4);
    #else
    _llk_pack_hw_configure_<UNTILIIZE, is_fp32_dest_acc_en>(DATA_FORMAT, DATA_FORMAT, 16*16*4);
    #endif

    _llk_pack_init_<UNTILIIZE, false, DstTileFaceLayout::RowMajor, false>(DATA_FORMAT);
    
    #ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncFull,DstTileFaceLayout::RowMajor,is_fp32_dest_acc_en>();
    #else
    _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, UNTILIIZE, is_fp32_dest_acc_en>();
    #endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull,UNTILIIZE, is_fp32_dest_acc_en>(0, L1_ADDRESS(buffer_Dest));
    _llk_pack_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
}

#endif
