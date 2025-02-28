// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "llk_defs.h"
#include "ckernel.h"

const bool unpack_to_dest = true;

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

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"


void run_kernel()
{
    volatile uint32_t* buffer_A = (volatile uint32_t*)0x1a000;

    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(DATA_FORMAT,DATA_FORMAT,FACE_R_DIM,0,4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, DATA_FORMAT,DATA_FORMAT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_A), 0, DATA_FORMAT,DATA_FORMAT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_common.h"
#include "ckernel_sfpu.h"
#include "ckernel_sfpu_binary.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

#define ELTWISE_BINARY_SFPU_OP 0

void run_kernel()
{
    const bool is_int_fpu_en = false;
    // copy srca to dest
    #ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE,false, is_fp32_dest_acc_en, is_int_fpu_en>(0, 0, 4, DATA_FORMAT);
    #else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE, is_fp32_dest_acc_en, is_int_fpu_en>(0, 0, 4, DATA_FORMAT);
    #endif
    _llk_math_pack_sync_init_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT, DATA_FORMAT);

    // copy srcA 
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, is_fp32_dest_acc_en, unpack_to_dest>(0, DATA_FORMAT, DATA_FORMAT);

    // copy srcB
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, is_fp32_dest_acc_en, unpack_to_dest>(1, DATA_FORMAT, DATA_FORMAT);

    _sfpu_binary_init_<false,ELTWISE_BINARY_SFPU_OP>();
    _llk_math_eltwise_binary_sfpu_start_<>();
    // compute

        for (int face = 0; face < 4; face++) {
            //sfpu_func(dst_offset, static_cast<ARGS&&>(args)...); -> sfpu add
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }


    _llk_math_eltwise_binary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
}

#endif 

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"


void run_kernel()
{

    volatile uint32_t* buffer_Dest = (volatile uint32_t*)0x1c000;

    std::fill(buffer_Dest, buffer_Dest + 16 * 16 * 4, 0xdeadbeef);

    #ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<false, is_fp32_dest_acc_en, false>(DATA_FORMAT, PACK_DEST_FORMAT, 16*16);
    #else
    _llk_pack_hw_configure_<false, is_fp32_dest_acc_en>(DATA_FORMAT, PACK_DEST_FORMAT, 16*16);
    #endif

    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(PACK_DEST_FORMAT);
    
    #ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncFull,DstTileFaceLayout::RowMajor,is_fp32_dest_acc_en>();
    #else
    _llk_pack_dest_init_<DstSync::SyncFull, DstTileFaceLayout::RowMajor, false, false>();
    #endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull,false, is_fp32_dest_acc_en>(0, L1_ADDRESS(buffer_Dest));
    _llk_pack_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
}

#endif
