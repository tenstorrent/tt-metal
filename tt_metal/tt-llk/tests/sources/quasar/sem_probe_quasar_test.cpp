// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// T0 probe (dest-semaphore design, not for commit). Dest handed over by Tensix semaphores in banks 1-3, no dvalids.
//   Pass 1: U unpacks -> F datacopy -> S (SFPU thread) square -> P packs tiles [0, N).            Proves bank_sel 1-3
//           SEMPOST/SEMWAIT/SEMGET work on every thread, the SFPU thread included.
//   Pass 2: F ZEROACC -> STALLWAIT(MATH) -> post to P -> P packs tiles [N, 2N), must read 0.    T0b: the clear lands
//           before the post.
// Every semaphore is back at 0 at kernel end.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

namespace probe
{
// id = 8 * bank + bit. Bit 4 avoids the perf barrier (ids 8-11).
constexpr std::uint32_t GO_S = 12; // F -> S, bank 1
constexpr std::uint32_t GO_P = 20; // S -> P and F -> P, bank 2
constexpr std::uint32_t GO_F = 28; // P -> F, bank 3

#define PROBE_POST(id)       TT_SEMPOST((id) >> 3, 1u << ((id) & 7))
#define PROBE_GET(id)        TT_SEMGET((id) >> 3, 1u << ((id) & 7))
#define PROBE_WAIT(res, id)  TT_SEMWAIT((res) | ckernel::p_stall::STALL_SYNC, ckernel::p_stall::STALL_ON_ZERO, (id) >> 3, 1u << ((id) & 7))
} // namespace probe

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_A             = params.buffer_A;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
#endif
    const auto bfd_unpack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, TILE_CNT);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
#endif
    const DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<false /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en>(src_format, src_format);
    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1 /*num_matrices*/);

    // Pass 1: F owns Dest at kernel start.
    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_(i);
    }
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU, p_stall::NOTHING, p_stall::MATH);
    PROBE_POST(probe::GO_S);

    // Pass 2: take Dest back from P, clear it, hand it straight to P.
    PROBE_WAIT(p_stall::STALL_MATH, probe::GO_F);
    PROBE_GET(probe::GO_F);
    TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_0, 0);
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::MATH);
    PROBE_POST(probe::GO_P);

    // Kernel end: park the token with F.
    PROBE_WAIT(p_stall::STALL_MATH, probe::GO_F);
    PROBE_GET(probe::GO_F);
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT = params.TILE_CNT;
#endif
    _llk_math_eltwise_sfpu_init_();
    init_square();

    PROBE_WAIT(p_stall::STALL_SFPU, probe::GO_S);
    PROBE_GET(probe::GO_S);
    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
    {
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_square, (SFPU_ITERATIONS), i, VectorMode::RC);
    }
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::WAIT_SFPU);
    PROBE_POST(probe::GO_P);
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_Res           = params.buffer_Res;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
#endif
    const auto bfd_pack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);

    for (std::uint32_t pass = 0; pass < 2; ++pass)
    {
        PROBE_WAIT(p_stall::STALL_PACK, probe::GO_P);
        PROBE_GET(probe::GO_P);
        for (std::uint32_t i = 0; i < TILE_CNT; ++i)
        {
            _llk_pack_(i /*dest_idx*/, pass * TILE_CNT + i, ckernel::DEFAULT_TENSOR_SHAPE);
        }
        TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::NOTHING, p_stall::NOTHING, p_stall::PACK0);
        PROBE_POST(probe::GO_F);
    }
}
#endif
