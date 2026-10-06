// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Drives the Quasar EMA entry (llk_math_ema_sfpu_entry.h) the way the ema compute kernel does:
// init, load the weights and clear the carry once, then feed TILE_CNT time tiles top to bottom with
// the carry chained through LREG4 from one tile to the next. Each input tile t is copied to Dest
// tile 2t and the entry writes its EMA to tile 2t + 1, so the production dst + 1 output store is
// what gets packed.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Input tile t sits at Dest tile EMA_DST_STRIDE * t; its EMA lands one tile further on.
constexpr std::uint32_t EMA_DST_STRIDE = 2;

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
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    }
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);

    // SrcA unpack is not a dest-dvalid client; do not inherit an UNP_DEST wait mask.
    set_up_zero_dest_dvalid_handshake_for_unpack();

    // Unpacks all TILE_CNT tiles into SrcA, one per math datacopy.
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

// The entry bounds-checks Dest against the compute kernel's DST_SYNC_MODE / DST_ACCUM_MODE.
constexpr ckernel::DstSync DST_SYNC_MODE = dest_sync;
constexpr bool DST_ACCUM_MODE            = is_fp32_dest_acc_en;

#include "llk_sfpu/llk_math_ema_sfpu_entry.h"

using namespace ckernel;
using namespace ckernel::math;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const DataFormat src_format = static_cast<DataFormat>(formats.math);

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(src_format, src_format);
    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(params.num_faces * params.TEST_FACE_R_DIM, 1 /*num_matrices*/);

    // One chain over the whole input: weights and a cleared carry once, before the first tile.
    llk_math_ema_sfpu_init();
    llk_math_ema_sfpu_load_alpha_beta(EMA_ALPHA_BITS, EMA_BETA_BITS);
    llk_math_ema_sfpu_clear_previous_output();

    set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();

    for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
    {
        _llk_math_eltwise_unary_datacopy_(EMA_DST_STRIDE * t);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();

    // Top to bottom: each call continues the carry the previous one left in LREG4.
    for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
    {
        llk_math_ema_sfpu_tile(EMA_DST_STRIDE * t);
    }
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

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
    // One tile per pack: the outputs sit at every other Dest tile.
    _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/);

    set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    for (std::uint32_t t = 0; t < params.TILE_CNT; ++t)
    {
        _llk_pack_(EMA_DST_STRIDE * t + 1 /*EMA output*/, t /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
    }
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
