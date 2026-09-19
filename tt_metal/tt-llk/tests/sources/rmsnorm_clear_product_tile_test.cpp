// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Seed every DEST slot, clear each product slot independently, and pack the whole
// section. Repeating an odd number of clears visits both half-sync banks for each
// target. The final logical slot is the reserved cross-chunk accumulator; any
// additional physical slots guard against clearing outside that capacity.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t cycle = 0; cycle < RMSNORM_CLEAR_CYCLES; ++cycle)
    {
        for (std::uint32_t target = 0; target < RMSNORM_CAPACITY - 1; ++target)
        {
            for (std::uint32_t tile = 0; tile < RMSNORM_DEST_TILES; ++tile)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

// Existing experimental broadcast definitions have unused parameters; keep their
// warning suppression local to the header, as in its broadcast-reuse driver.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(RMSNORM_DEST_TILES == get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>());
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_sfpu_init_once_();

    for (std::uint32_t cycle = 0; cycle < RMSNORM_CLEAR_CYCLES; ++cycle)
    {
        for (std::uint32_t target = 0; target < RMSNORM_CAPACITY - 1; ++target)
        {
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                TILE_NUM_FACES, formats.math);
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (std::uint32_t tile = 0; tile < RMSNORM_DEST_TILES; ++tile)
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    tile, formats.math, formats.math);
            }

            // Same raw init as compute::mul_reduce_scalar_init: it restores the
            // non-incrementing ADDR_MOD_1 required by the product clear.
            _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MathFidelity::HiFi4, EltwiseBinaryReuseDestType::NONE>(
                ckernel::DEFAULT_TENSOR_SHAPE, 0);

            // Leave a nonuniform SFPU write immediately before the clear, so its
            // WAIT_SFPU transition is exercised as well as ZEROACC addressing.
            _llk_math_eltwise_unary_sfpu_params_(
                []
                {
                    for (std::uint32_t row = 0; row < 8; ++row)
                    {
                        sfpi::vFloat value = sfpi::dst_reg[0];
                        sfpi::dst_reg[0]   = -value;
                        sfpi::dst_reg++;
                    }
                },
                target,
                VectorMode::RC);
            _llk_math_rmsnorm_clear_product_tile_<RMSNORM_CAPACITY, is_fp32_dest_acc_en>(target);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
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
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    std::uint32_t output = 0;
    for (std::uint32_t cycle = 0; cycle < RMSNORM_CLEAR_CYCLES; ++cycle)
    {
        for (std::uint32_t target = 0; target < RMSNORM_CAPACITY - 1; ++target)
        {
            _llk_packer_wait_for_math_done_();
            for (std::uint32_t tile = 0; tile < RMSNORM_DEST_TILES; ++tile)
            {
                _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[output++]));
            }
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif
