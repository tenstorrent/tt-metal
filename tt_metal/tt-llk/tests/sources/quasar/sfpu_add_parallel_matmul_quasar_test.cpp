// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Parallel FPU matmul (TRISC0-2 -> buffer_C) + isolated SrcS SFPU add (TRISC3 -> buffer_Res).
// FPU path uses Dest dvalid {FPU, PACK}. SFPU path uses UNP_S/SrcS/PACK1 only (no Dest).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);

        // Matmul flips the unpacker roles: _llk_unpack_matmul_init_ arg0 drives UNPACR1/SrcB, arg1 drives
        // UNPACR0/SrcA -- so operand A is recorded under Unp1 and operand B under Unp0 (matches product).
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
            ckernel::tensor_shape_from_num_faces(FACE_R_DIM, params.num_faces_A), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(FACE_R_DIM, params.num_faces_B), L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src);
        _llk_unpack_configure_binary_<p_unpacr::UNP_B, p_unpacr::UNP_A>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));

        _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            CT_DIM,
            RT_DIM,
            KT_DIM);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    _llk_unpack_matmul_(CT_DIM, RT_DIM, KT_DIM, j, j * CT_DIM);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*EN_INT32_MATH_FORMAT*/>(
            static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));
        _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_DIRECT_INDEXING, ENABLE_2X_FORMAT>(CT_DIM, RT_DIM);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < KT_DIM; i++)
                {
                    _llk_math_matmul_block_(CT_DIM, RT_DIM);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < KT_DIM; i++)
                {
                    _llk_math_matmul_block_(CT_DIM, RT_DIM);
                }
                _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_binary.h"
#include "llk_sfpu/ckernel_sfpu_srcs.h"
#include "llk_sfpu_srcs_api.h"
#include "llk_srcs.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const volatile FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles   = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;

    const DataFormat srcs_format = static_cast<DataFormat>(formats.unpack_S_dst);

    {
        ZONE_SCOPED("INIT")
        // The SFPI slice kernel uses the same layout for its input and output operands.
        LLK_ASSERT(srcs_format == static_cast<DataFormat>(formats.pack_S_src), "SrcS ADD requires matching unpack destination and pack source formats");
        // FormatConfig has no unpack_T_* fields, so T is unpacked with S's format. Callers must keep
        // stimuli_T_format == stimuli_S_format or T will be read with the wrong format.
        llk_sfpu_srcs_binary_init(
            L1_ADDRESS(params.buffer_S[0]),
            L1_ADDRESS(params.buffer_T[0]),
            static_cast<DataFormat>(formats.unpack_S_src),
            static_cast<DataFormat>(formats.unpack_S_dst),
            L1_ADDRESS(params.buffer_Res[0]),
            static_cast<DataFormat>(formats.pack_S_src),
            static_cast<DataFormat>(formats.pack_S_dst),
            IMPLIED_MATH_FORMAT);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        // SFPU_ISOLATE = the whole TRISC3 SrcS pipeline (UNP_S -> SFPU -> PACK1) without the matmul TRISCs.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
            dispatch_sfpu_srcs_format(
                srcs_format,
                [&](auto format)
                {
                    using Format = decltype(format);
                    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                    {
                        llk_sfpu_srcs_binary(num_tiles, srcs_format, [](int, int, int, int) { calculate_add_srcs<Format::layout>(); });
                    }
                });
        }
        wait_unpack_idle();
        wait_sfpu_idle();
        wait_pack_idle();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        // PACK_ISOLATE and L1_CONGESTION pack without a math<->pack handshake.
        // Explicitly clear wait_mask because CFG state can persist across run types.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_C[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_matmul_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), RT_DIM, CT_DIM, 1 /*num_subblocks_c_dim*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::SFPU_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/);
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
