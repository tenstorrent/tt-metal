// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <algorithm>

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
    volatile uint32_t* const buffer_A = reinterpret_cast<volatile uint32_t*>(0x1a000);

    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(DATA_FORMAT,DATA_FORMAT,FACE_R_DIM,0,4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, DATA_FORMAT,DATA_FORMAT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_A), 0, DATA_FORMAT,DATA_FORMAT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_eltwise_unary_datacopy.h"
#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel()
{
    // copy srca to dest
    #ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE,false, is_fp32_dest_acc_en, false>(0, 0, 4, DATA_FORMAT);
    #else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, BroadcastType::NONE, is_fp32_dest_acc_en, false>(0, 0, 4, DATA_FORMAT);
    #endif
    _llk_math_pack_sync_init_<DstSync::SyncFull,is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false,false>(DATA_FORMAT, DATA_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncFull, BroadcastType::NONE, is_fp32_dest_acc_en, unpack_to_dest>(0, DATA_FORMAT, DATA_FORMAT);
    
    // calculation of sfpu operation on dest
    _llk_math_eltwise_unary_sfpu_init_<SFPU_OPERATION>();
    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncFull>(0);
    // calling sfpu function from ckernel
    // this part is where parametrization of operation takes part
    #ifdef SFPU_CALLS
    SFPU_CALLS
    #endif

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncFull,is_fp32_dest_acc_en>();
}

#endif 

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

// If data foramt is Bfp8 it is calculated correctly in Dest but packer cannot pack just that one face
// TODO: make it so It can
// So for now It is packed as Float16_b 

#ifdef FORMAT_BFP8_B
#define PACK_DEST_FORMAT (uint32_t)DataFormat::Float16_b
#else
#define PACK_DEST_FORMAT DATA_FORMAT
#endif


void run_kernel()
{

    volatile uint32_t* const buffer_Dest = reinterpret_cast<volatile uint32_t*>(0x1c000);

    std::fill(buffer_Dest, buffer_Dest + 16 * 16 * 4, 0xdeadbeef);

    #ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<false, is_fp32_dest_acc_en, false>(DATA_FORMAT, PACK_DEST_FORMAT, 16*16*4);
    #else
    _llk_pack_hw_configure_<false, is_fp32_dest_acc_en>(DATA_FORMAT, PACK_DEST_FORMAT, 16*16*4);
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
