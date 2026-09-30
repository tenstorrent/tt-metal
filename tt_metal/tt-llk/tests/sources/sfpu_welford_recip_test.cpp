// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exactness check of the reciprocal the Welford SFPU kernel computes when it is given no table
// (_load_recip_of_idx_<0> in sfpu/ckernel_sfpu_welfords.h). For every idx in
// [WELFORD_RECIP_BASE, WELFORD_RECIP_BASE + 32 * TILE_CNT) the math thread runs that load, which
// leaves the lane-uniform fp32 reciprocal 1 / (idx + 1) in LREG7, and stores the vector into one
// 4-row by 8-column slab of DEST tile 0: slab s of result tile t holds 1 / (WELFORD_RECIP_BASE + 32 t + s + 1)
// in all 32 of its elements. The pack thread packs each tile as Float32 out of a 32-bit DEST, so the
// host can compare the bit pattern with its own fp32 division (test_sfpu_welford.py). The run needs
// dest_acc Yes and a Float32 output; no input tile is unpacked.

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

static constexpr std::uint32_t RECIP_DST_INDEX = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Nothing is unpacked: the SFPU writes DEST directly. The configuration keeps the thread's state
    // consistent with the other two.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"

using namespace ckernel;

// The four slabs of a 4-row group: even columns of the left face, odd columns of the left face, even and odd
// columns of the right face (the offsets the Welford and EMA bodies use for one block).
static constexpr std::uint32_t SLAB_OFFSET[4] = {0, 2, 16, 18};

static const std::array<std::uint32_t, 0> no_lut {};

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Welford init: the SFPU configuration (the programmable constants the reciprocal uses) and ADDR_MOD_7.
    _llk_math_welfords_sfpu_init_();

    std::uint32_t idx = WELFORD_RECIP_BASE;
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        _llk_math_eltwise_sfpu_start_(RECIP_DST_INDEX);
        for (std::uint32_t slab = 0; slab < 32; ++slab)
        {
            // Slab s: face pair s / 16, 4-row group (s / 4) % 4, column half and face s % 4.
            const std::uint32_t offset = 32 * (slab >> 4) + 4 * ((slab >> 2) & 3) + SLAB_OFFSET[slab & 3];
            _load_recip_of_idx_<0>(idx, no_lut);
            TT_SFPSTORE(ckernel::p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ckernel::ADDR_MOD_7, offset);
            ++idx;
        }
        _llk_math_eltwise_sfpu_done_();
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(RECIP_DST_INDEX, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
