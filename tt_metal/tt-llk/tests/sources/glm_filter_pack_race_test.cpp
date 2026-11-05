// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single-core repro of the tt-blaze glm_sparse_read_sdpa tilizer index filter (tt-blaze#6224).
//
// One UInt32 index tile runs through the same window as the tilizer, in a kernel compiled for a 16-bit
// DEST (is_fp32_dest_acc_en = false) with SyncFull:
//   1. optional: MATH sets DEST_ACCESS_CFG remap_addrs + swizzle_32b, as hw_cleanup / fast tilize leave it;
//   2. all threads switch DEST to fp32 (_llk_set_fp32_dest_acc_);
//   3. PACK reconfigures for the UInt32 mask and re-inits (pack_reconfig_data_format<true>);
//   4. UNPACK unpacks the tile to DEST; MATH copies it, runs the SFPU bank filter, then the topk_xl
//      K=1024 per-column sort, and commits;
//   5. PACK waits, optionally delays, optionally re-runs the per-tile packer reconfig, and packs;
//   6. all threads switch DEST back.
//
// Template knobs (params.h): RS_REMAP, RS_SORT, RS_PER_TILE_RECONFIG, RS_PACK_DELAY (RISC nops on PACK
// between the MATH handoff and the pack) and, when RS_DRAIN is set, RS_DRAIN_BEFORE_RELEASE, which turns
// on the gated #55807 drain in this tree's llk_fp32_dest_acc.h.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncFull;

// The tilizer's bank filter constants (DSv4.1 decoder, tilizer of KV bank 6).
static constexpr std::uint32_t RS_BANK_MASK        = 15;
static constexpr std::uint32_t RS_MY_BANK          = 6;
static constexpr std::uint32_t RS_GLOBAL_BANK_SHIFT = 14;
static constexpr std::uint32_t RS_WITHIN_BANK_MASK = 16383;

static inline void rs_delay(std::uint32_t n)
{
    for (std::uint32_t i = 0; i < n; ++i)
    {
        asm volatile("nop");
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_fp32_dest_acc.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    for (std::uint32_t rep = 0; rep < RS_REPS; ++rep)
    {
        _llk_set_fp32_dest_acc_<ckernel::ThreadId::UnpackThreadId>();
        rs_delay(RS_UNPACK_DELAY);  // the tilizer's cb_wait_front(indices) on late NOC data

        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true /* unpack_to_dest */>(
            0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true /* unpack_to_dest */>(
            L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

        if constexpr (RS_TAIL)
        {
            // tensix_sync + get_tile_address(filter_mask, 0): UNPACK hands the mask address to MATH and PACK.
            tensix_sync();
            mailbox_write(ckernel::ThreadId::MathThreadId, L1_ADDRESS(params.buffer_Res[0]));
            mailbox_write(ckernel::ThreadId::PackThreadId, L1_ADDRESS(params.buffer_Res[0]));
        }
        _llk_set_fp32_dest_acc_<ckernel::ThreadId::UnpackThreadId>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_fp32_dest_acc.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_sparse_k_filter.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        4, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    if constexpr (RS_REMAP)
    {
        // What fast tilize, pack_untilize and compute_kernel_hw_cleanup leave behind.
        _llk_math_reconfig_remap_(true);
    }

    for (std::uint32_t rep = 0; rep < RS_REPS; ++rep)
    {
    _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(true);

    // copy_init + binary_bitwise_tile_init in the tilizer: datacopy MOP, then SFPU address modifiers.
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        4, formats.math);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    rs_delay(RS_MATH_DELAY);
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, true /* unpack_to_dest */>(
        0, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_sparse_k_filter_tile_<32, RS_BANK_MASK, RS_MY_BANK, RS_GLOBAL_BANK_SHIFT, RS_WITHIN_BANK_MASK, 0>,
        0,
        VectorMode::RC_custom);
    if constexpr (RS_SORT)
    {
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::_topk_xl_init_<1024, true>();
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::_topk_xl_local_sort_generic_<1024, true>, 0, VectorMode::RC_custom, 0u, false);
    }
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    if constexpr (RS_TAIL)
    {
        tensix_sync();
        volatile std::uint32_t rs_addr = mailbox_read(ThreadId::UnpackThreadId);
        (void)rs_addr;
    }
    _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(false);
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_fp32_dest_acc.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr std::uint32_t tile_size = FACE_R_DIM * FACE_C_DIM * 4;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    for (std::uint32_t rep = 0; rep < RS_REPS; ++rep)
    {
    _llk_set_fp32_dest_acc_<ckernel::ThreadId::PackThreadId>();

    // pack_reconfig_data_format<true>(filter_mask) at the tilizer's window entry.
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, tile_size);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);

    _llk_packer_wait_for_math_done_();
    rs_delay(RS_PACK_DELAY);
    if constexpr (RS_PER_TILE_RECONFIG)
    {
        // pack_reconfig_data_format(filter_mask) inside glm_sparse_filter_mask_tile.
        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, tile_size);
    }
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    if constexpr (RS_TAIL)
    {
        tensix_sync();
        volatile std::uint32_t rs_addr = mailbox_read(ckernel::ThreadId::UnpackThreadId);
        (void)rs_addr;
    }
    _llk_set_fp32_dest_acc_<ckernel::ThreadId::PackThreadId>();
    }
}

#endif
