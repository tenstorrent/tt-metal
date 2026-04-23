// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Replicates the exact ttnn tilize compute kernel flow:
//   compute_kernel_hw_startup → fast_tilize_init → fast_tilize_block × N → fast_tilize_uninit
// Validates that BH fast-tilize LLK works after compute_kernel_hw_startup.
//
// Row-looping reference: block height is 1; the outer row loop is the caller's
// responsibility. This source exercises BLOCK_RT_DIM > 1 via that outer loop,
// serving as the multi-row composition integration test.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t MAX_UNITS = 16;

inline std::uint32_t decompose_row(const std::uint32_t ct_dim, std::uint32_t unit_dims[MAX_UNITS])
{
    std::uint32_t n4 = ct_dim / 4, rem = ct_dim % 4, idx = 0;
    if (rem == 1 && n4 > 0)
    {
        n4--;
        rem = 5;
    }
    for (std::uint32_t i = 0; i < n4; i++)
    {
        unit_dims[idx++] = 4;
    }
    if (rem == 2)
    {
        unit_dims[idx++] = 2;
    }
    else if (rem == 3)
    {
        unit_dims[idx++] = 3;
    }
    else if (rem == 5)
    {
        unit_dims[idx++] = 2;
        unit_dims[idx++] = 3;
    }
    return idx;
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const Operand& buffer_A          = params.buffer_A;
#endif

    // compute_kernel_hw_startup: llk_unpack_hw_configure
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    std::uint32_t unit_dims[MAX_UNITS];
    std::uint32_t units_per_row = decompose_row(BLOCK_CT_DIM, unit_dims);

    // fast_tilize_init: X configured for first chunk; no reinit needed on first call.
    const std::uint32_t first_chunk = BLOCK_CT_DIM > 5 ? 4 : BLOCK_CT_DIM == 5 ? 2 : BLOCK_CT_DIM;
    _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, BLOCK_CT_DIM, first_chunk);

    // fast_tilize_block × rows: caller loops rows, one block call per chunk per row.
    std::uint32_t prev_chunk = first_chunk;
    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
    {
        for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
        {
            std::uint32_t col_offset = 0;
            for (std::uint32_t u = 0; u < units_per_row; u++)
            {
                std::uint32_t chunk = unit_dims[u];
                if (chunk != prev_chunk)
                {
                    _llk_unpack_fast_tilize_reinit_xdim_(chunk);
                    prev_chunk = chunk;
                }
                _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[row * BLOCK_CT_DIM]), 0, formats.unpack_A_src, chunk, 4, col_offset);
                col_offset += chunk;
            }
        }
    }

    // fast_tilize_uninit
    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
#endif

    std::uint32_t unit_dims[MAX_UNITS];
    std::uint32_t units_per_row = decompose_row(BLOCK_CT_DIM, unit_dims);

    // compute_kernel_hw_startup: sync_init + hw_configure
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // fast_tilize_init
    _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(formats.math);

    // fast_tilize_block × rows (one section_done per chunk)
    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
    {
        for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
        {
            for (std::uint32_t u = 0; u < units_per_row; u++)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(0, formats.math, 4);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
    }

    // fast_tilize_uninit
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_fast_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const Operand& buffer_Res        = params.buffer_Res;
#endif

    std::uint32_t unit_dims[MAX_UNITS];
    std::uint32_t units_per_row = decompose_row(BLOCK_CT_DIM, unit_dims);

    // Match EXACT order of working LLK test: dest_init → hw_configure
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));

    // fast_tilize_init
    _llk_pack_fast_tilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, formats.pack_dst, unit_dims[0], 4);

    // Row-scoped pack: destination programmed once per row, chunks streamed through.
    std::uint32_t prev_udim = unit_dims[0];
    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
    {
        for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
        {
            _llk_pack_fast_tilize_row_begin_(L1_ADDRESS(buffer_Res[row * BLOCK_CT_DIM]));

            for (std::uint32_t u = 0; u < units_per_row; u++)
            {
                std::uint32_t udim = unit_dims[u];
                if (udim != prev_udim)
                {
                    _llk_pack_fast_tilize_reinit_unit_dim_(formats.pack_dst, udim);
                    prev_udim = udim;
                }
                _llk_packer_wait_for_math_done_();
                _llk_pack_fast_tilize_row_chunk_(0, udim, 4);
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }

            _llk_pack_fast_tilize_row_end_();
        }
    }

    // fast_tilize_uninit
    _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, FACE_R_DIM, 4);
}

#endif
