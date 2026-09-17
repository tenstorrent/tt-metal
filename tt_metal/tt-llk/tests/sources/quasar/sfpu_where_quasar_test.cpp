// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Buffer layout: three input tiles in buffer_A (cond, true_val, false_val) are
// staged into Dest[DST_INDEX + 0/1/2]. SFPU where writes the result to
// Dest[DST_INDEX]; PACK reads that one tile into buffer_Res.
//
// Two execution paths, selected by `unpack_to_dest`:
//   * unpack_to_dest=true  — UNPACK writes Dest; SFPU reads/writes Dest; PACK reads Dest.
//   * unpack_to_dest=false — UNPACK -> SrcA -> FPU datacopy -> Dest; SFPU; PACK.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
#endif
    LLK_ASSERT(TILE_CNT == 3, "Where stages three Dest tiles (cond, true, false)");

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);

        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            // If Dst is 32b and MATH uses FPU datacopy (MOVA2D → ELWADD fallback), we need both SrcA and SrcB formats configured.
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        }

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, TILE_CNT);

        // Program dest-dvalid CFG after HW configure so wait masks are the last
        // writes before TILE_LOOP. UNPACK is only a dest client on UNP_DEST.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
            else
            {
                // SrcA unpack is not a dest-dvalid client; do not inherit an
                // UNP_DEST wait mask from a prior kernel on this device.
                set_up_zero_dest_dvalid_handshake_for_unpack();
            }
        }
        else
        {
            // CFG wait masks persist across run types on the same device.
            // Isolated/congested UNPACK must not inherit the L1_TO_L1 chain.
            set_up_zero_dest_dvalid_handshake_for_unpack();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                // Quasar's 32-bit A2D datacopy is ELWADD and consumes SrcA
                // plus the dummy SrcB dvalid emitted by the unpack MOP.
                _perf_unpack_loop_set_valid<true /*set_a*/, is_fp32_dest_acc_en>(LOOP_FACTOR * TILE_CNT);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
                if constexpr (unpack_to_dest && PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_sfpu/ckernel_sfpu_where.h"
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t DST_INDEX       = params.DST_INDEX;
#endif
    LLK_ASSERT(TILE_CNT == 3, "Where stages three Dest tiles (cond, true, false)");

    DataFormat src_format = static_cast<DataFormat>(formats.math);

    {
        ZONE_SCOPED("INIT")
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

        if constexpr (!unpack_to_dest)
        {
            _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1 /*num_matrices*/);
        }
        // SFPU uses ADDR_MOD_7 and does not overwrite the datacopy MOP's
        // ADDR_MOD_0/1 or bank-0 programming, so both initializers can remain in
        // the INIT zone before the measured TILE_LOOP.
        _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();

        // Program dest-dvalid CFG after HW configure so wait masks are the last
        // writes before TILE_LOOP.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            // The dvalid chain must match UNPACK exactly. On the FPU path T1 is
            // both the datacopy producer and the SFPU producer.
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
            }
            else
            {
                set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
                set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
            }
        }
        else
        {
            // CFG wait masks persist across run types. MATH_ISOLATE runs without
            // a pack consumer; UNPACK_ISOLATE / L1_CONGESTION only mock Src
            // handshakes. None of them may inherit the FPU→SFPU→PACK chain.
            set_up_zero_dest_dvalid_handshake_for_math();
            set_up_zero_dest_dvalid_handshake_for_sfpu();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, is_fp32_dest_acc_en>(LOOP_FACTOR * TILE_CNT);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (!unpack_to_dest)
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(DST_INDEX + i);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }

                // Runs calculate_where over the faces selected by VECTOR_MODE: cond=base+0,
                // true_val=base+1, false_val=base+2, result written to base+0. Faces
                // outside the selected set keep whatever the producer wrote into Dest before
                // SFPU ran (the cond tile, here), so Python asserts only processed faces.
                SFPU_TERNARY_CALL(
                    dest_sync,
                    is_fp32_dest_acc_en,
                    calculate_where,
                    (false /*APPROXIMATION_MODE*/),
                    DST_INDEX + 0u /*DST_IN0*/,
                    DST_INDEX + 1u /*DST_IN1*/,
                    DST_INDEX + 2u /*DST_IN2*/,
                    DST_INDEX + 0u /*DST_OUT*/,
                    VECTOR_MODE);

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
                }
            }
            // Drain every execution unit driven by T1 before PACK takes over.
            wait_sfpu_idle();
            wait_fpu_idle();
            wait_mop_idle();
        }
        PROFILER_SYNC();
    }
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t DST_INDEX   = params.DST_INDEX;
    const Operand& buffer_Res       = params.buffer_Res;
#endif

    {
        ZONE_SCOPED("INIT")
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);

        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*only the SFPU result tile*/);

        // Program dest-dvalid CFG after HW configure so wait masks are the last
        // writes before TILE_LOOP.
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            // Declare the same dvalid client chain that UNPACK and MATH use.
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
            else
            {
                set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
        }
        else
        {
            // PACK_ISOLATE / L1_CONGESTION pack independently. UNPACK_ISOLATE /
            // MATH_ISOLATE do not pack. None may inherit the L1_TO_L1 wait mask.
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_pack_(DST_INDEX, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    // Drain this bank's dest-waited pack before issuing the next.
                    // Queuing all LOOP_FACTOR packs into one tensix_sync makes the
                    // TILE_LOOP ZONE_END wall-clock read return 0 on long SFPU ops.
                    ckernel::wait_pack_idle();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
