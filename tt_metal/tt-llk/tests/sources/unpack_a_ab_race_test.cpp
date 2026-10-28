// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Race repro: per DST section, datacopy (unpack_A) slots 0..3, a math-only delay,
// datacopy slots 4..5, then ELWADD (unpack_AB) slots 6..7. On the first run after
// tt-smi -r, slot 6 of the first section reads SrcB as zero. DELAY_CYCLES = 0 passes.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

using namespace ckernel;
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t COPIES_BEFORE = 4;
constexpr std::uint32_t COPIES_AFTER  = 6;
constexpr std::uint32_t DELAY_CYCLES  = 300;

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<false>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            false /* transpose_of_faces */, false /* within_face_16x16_transpose */, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        for (std::uint32_t slot = 0; slot < COPIES_AFTER; ++slot)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                L1_ADDRESS(params.buffer_A[base + slot]), formats.unpack_A_src, formats.unpack_A_dst);
        }
        _llk_unpack_AB_init_<BroadcastType::NONE>(DEFAULT_TENSOR_SHAPE, ckernel::Transpose::None);
        for (std::uint32_t slot = COPIES_AFTER; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[base + slot]), L1_ADDRESS(params.buffer_B[base + slot]));
        }
    }
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<false>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, false, BroadcastType::NONE, false, PackMode::Default>(TILE_NUM_FACES, formats.math);
        for (std::uint32_t slot = 0; slot < COPIES_BEFORE; ++slot)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, false, BroadcastType::NONE, false>(slot, formats.math, formats.math);
        }
        // Math falls behind the unpacker here (any slow math-only op does the same).
        tensix_sync();
        ckernel::wait(DELAY_CYCLES);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, false, BroadcastType::NONE, false, PackMode::Default>(TILE_NUM_FACES, formats.math);
        for (std::uint32_t slot = COPIES_BEFORE; slot < COPIES_AFTER; ++slot)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, false, BroadcastType::NONE, false>(slot, formats.math, formats.math);
        }
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(DEFAULT_TENSOR_SHAPE, 0 /* acc_to_dest */);
        for (std::uint32_t slot = COPIES_AFTER; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, DstSync::SyncHalf, false>(DEFAULT_TENSOR_SHAPE, slot);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
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
    _llk_pack_hw_configure_<false, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_R_DIM * TILE_C_DIM /* tile_size */);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, false, PackMode::Default>();
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    for (std::uint32_t base = 0; base < params.TILE_CNT; base += DEST_NUM_TILES_FP16_HALF)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t slot = 0; slot < DEST_NUM_TILES_FP16_HALF; ++slot)
        {
            _llk_pack_<DstSync::SyncHalf, false, PackMode::Default>(slot, L1_ADDRESS(params.buffer_Res[base + slot]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    }
}
#endif
