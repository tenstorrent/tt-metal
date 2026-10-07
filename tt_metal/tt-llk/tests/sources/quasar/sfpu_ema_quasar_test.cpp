// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One EMA chain over TILE_CNT tiles via the entry. SyncFull: one Dest section for all tiles. SyncHalf:
// one section per tile, as ema_compute.cpp; MATH_TRANSPOSE_FACES adds transpose_dest between tiles.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

using namespace ckernel;
#include "params.h" // dest_sync, MATH_TRANSPOSE_FACES

// Mirrors sfpu::EMA_OUTPUT_TILE_DELTA (static_assert on the math thread).
constexpr std::uint32_t EMA_OUT_TILE_OFFSET = 1;
constexpr std::uint32_t EMA_DST_STRIDE      = EMA_OUT_TILE_OFFSET + 1;
constexpr std::uint32_t EMA_SECTION_IN_TILE = 0;

constexpr bool EMA_TILE_PER_SECTION = (dest_sync == ckernel::DstSync::SyncHalf);
static_assert(EMA_TILE_PER_SECTION || !MATH_TRANSPOSE_FACES, "the transpose_dest interleave runs with one tile per section");

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const auto bfd_unpack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);

    if constexpr (is_fp32_dest_acc_en)
    {
        // 32-bit A2D datacopy is ELWADD, so SrcB's format must be configured too.
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    }

    // SrcA unpack is not a dest-dvalid client; do not inherit an UNP_DEST wait mask.
    set_up_zero_dest_dvalid_handshake_for_unpack();

    if constexpr (EMA_TILE_PER_SECTION)
    {
        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/);
        for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
        {
            _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(t /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            if constexpr (MATH_TRANSPOSE_FACES)
            {
                // transpose_dest's MOVD2B reads stall on SrcB validity.
                _llk_unpack_set_srcB_dummy_valid_();
            }
        }
    }
    else
    {
        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
        _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_transpose_dest.h"
#include "params.h"

// The entry reads the compute kernel's DST_SYNC_MODE / DST_ACCUM_MODE.
constexpr ckernel::DstSync DST_SYNC_MODE = dest_sync;
constexpr bool DST_ACCUM_MODE            = is_fp32_dest_acc_en;

#include "llk_sfpu/llk_math_ema_sfpu_entry.h"

using namespace ckernel;
using namespace ckernel::math;

static_assert(EMA_OUT_TILE_OFFSET == sfpu::EMA_OUTPUT_TILE_DELTA, "the test's output offset must match the entry's");

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const DataFormat src_format   = static_cast<DataFormat>(formats.math);
    const std::uint32_t num_rows  = params.num_faces * params.TEST_FACE_R_DIM;
    constexpr bool interleave_fpu = MATH_TRANSPOSE_FACES;

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(src_format, src_format);
    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1 /*num_matrices*/);

    llk_math_ema_sfpu_init();
    llk_math_ema_sfpu_load_alpha_beta(EMA_ALPHA_BITS, EMA_BETA_BITS);
    llk_math_ema_sfpu_clear_previous_output();

    set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();

    if constexpr (EMA_TILE_PER_SECTION)
    {
        for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
        {
            if constexpr (interleave_fpu)
            {
                // Datacopy and transpose_dest share bank 0's MOP, so each re-inits before running.
                // EMA's bank-1 body, weights and carry survive both: no EMA re-init.
                _configure_default_alu_data_format_state_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(src_format, src_format);
                _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1 /*num_matrices*/);
            }
            _llk_math_eltwise_unary_datacopy_(EMA_SECTION_IN_TILE);
            if constexpr (interleave_fpu)
            {
                _configure_mov_ops_explicit_alu_data_format_state_<is_fp32_dest_acc_en>(src_format, src_format);
                _llk_math_transpose_dest_init_<true /*TRANSPOSE_OF_FACES*/, is_fp32_dest_acc_en>();
                _llk_math_transpose_dest_(EMA_SECTION_IN_TILE);
            }
            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();

            llk_math_ema_sfpu_tile(EMA_SECTION_IN_TILE);
            _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
        }
    }
    else
    {
        for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
        {
            _llk_math_eltwise_unary_datacopy_(EMA_DST_STRIDE * t);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();

        for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
        {
            llk_math_ema_sfpu_tile(EMA_DST_STRIDE * t);
        }
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    }

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const auto bfd_pack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    // One tile per pack: outputs are interleaved with inputs in Dest.
    _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/);

    set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    if constexpr (EMA_TILE_PER_SECTION)
    {
        for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
        {
            _llk_pack_(EMA_SECTION_IN_TILE + EMA_OUT_TILE_OFFSET /*start_math_dest_tile_idx*/, t /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            // Drain this section's pack before releasing the next.
            ckernel::wait_pack_idle();
        }
    }
    else
    {
        for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
        {
            _llk_pack_(EMA_DST_STRIDE * t + EMA_OUT_TILE_OFFSET /*start_math_dest_tile_idx*/, t /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
