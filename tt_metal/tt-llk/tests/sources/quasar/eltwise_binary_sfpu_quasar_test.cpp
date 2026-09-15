// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Buffer layout (params.TILE_CNT input tiles in buffer_A):
//   buffer_A[i] -> Dest[i]; SFPU writes the result to Dest[DST_TILE_IDX];
//   PACK reads Dest[DST_TILE_IDX] (one tile).
// For max/min the convention is in0=Dest[0], in1=Dest[1], out=Dest[2].
//
// Two execution paths, selected by `unpack_to_dest`:
//   * unpack_to_dest=true  — UNPACK writes Dest directly (32-bit Dest formats,
//                            and all integer/float mul/div variants).
//   * unpack_to_dest=false — UNPACK -> SrcA -> FPU datacopy -> Dest (required for
//                            non-32-bit / MX block formats used by max/min).

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

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            // DEST DVALID handshake: T0 is the producer. The producer differs
            // per path: UNPACK for UNP_DEST and FPU for the datacopy path.
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
            else
            {
                set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
        }
        else if constexpr (unpack_to_dest)
        {
            // L1_TO_L1 may leave UNP_DEST waiting for inactive SFPU/PACK clients.
            // Isolated/congested UNP_DEST execution must not inherit that chain.
            set_up_zero_dest_dvalid_handshake_for_unpack();
        }

        if constexpr (unpack_to_dest)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);

        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        }

        _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, TILE_CNT);
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

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"
#include "sfpu_operations_quasar.h"

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
    const std::uint32_t SRC0_TILE_IDX   = params.SRC0_TILE_IDX;
    const std::uint32_t SRC1_TILE_IDX   = params.SRC1_TILE_IDX;
    const std::uint32_t DST_TILE_IDX    = params.DST_TILE_IDX;
#endif
    // Binary SFPU stages both operands from buffer_A into Dest. TILE_CNT is the
    // number of staged tiles and is fixed by the test layout, not free-running:
    // 2 when operands occupy Dest tiles 0/1 (result in-place or a separately
    // written out tile, e.g. max/min's tile 2), or 4 for the result-aliasing
    // variant that places operands at Dest tiles 2/3. Guard against any other count.
    LLK_ASSERT(TILE_CNT == 2 || TILE_CNT == 4, "Binary SFPU expects TILE_CNT of 2 or 4");
    const DataFormat math_format = static_cast<DataFormat>(formats.math);

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            // DEST DVALID handshake. On the FPU path T1 is both the datacopy
            // producer and the SFPU producer, so both clients are registered.
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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // FPU datacopy (when present) and SFPU execute sequentially on this
            // thread. Keep them on one Dest bank instead of creating a dvalid
            // chain whose producer would also wait for inactive UNPACK/PACK.
            set_up_zero_dest_dvalid_handshake_for_math();
            set_up_zero_dest_dvalid_handshake_for_sfpu();
        }
        // Max/min selects the Dest mode (fp32 / int32 / neither) from pack_src
        // so integer ordering is preserved. Other ops use the inferred mode.
        if constexpr (test_utils::quasar_binary_op_is_max_min(SFPU_BINARY_OP))
        {
            configure_math_hardware_for_float32_int32_or_default<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(
                math_format, static_cast<DataFormat>(formats.pack_src));
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>(math_format, math_format);
        }

        if constexpr (!unpack_to_dest)
        {
            _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1 /*num_matrices*/);
        }
        // SFPU uses ADDR_MOD_7 and does not overwrite the datacopy MOP's
        // ADDR_MOD_0/1 or bank-0 programming, so both initializers can remain in
        // the INIT zone before the measured TILE_LOOP.
        _llk_math_eltwise_sfpu_init_();
        test_utils::init_binary_sfpu_operation_quasar<SFPU_BINARY_OP, is_fp32_dest_acc_en, SFPU_SIGN_MAGNITUDE, APPROX_MODE>(params.ZERO_POINT);
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
                test_utils::call_binary_sfpu_operation_quasar<
                    SFPU_BINARY_OP,
                    dest_sync,
                    is_fp32_dest_acc_en,
                    SFPU_DST_ROUNDING_MODE,
                    SFPU_ITERATIONS,
                    SFPU_SIGN_MAGNITUDE,
                    APPROX_MODE>(SRC0_TILE_IDX, SRC1_TILE_IDX, DST_TILE_IDX, math_format);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
                }
            }
            // Drain SFPU, FPU, and MOP queues before PACK takes over.
            wait_sfpu_idle();
            wait_fpu_idle();
            wait_mop_idle();
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

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
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t DST_TILE_IDX = params.DST_TILE_IDX;
    const Operand& buffer_Res        = params.buffer_Res;
#endif

    {
        ZONE_SCOPED("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // PACK runs without the L1_TO_L1 producer chain in isolated modes.
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
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

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);

        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*only the SFPU result tile*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_pack_(DST_TILE_IDX, 0 /*tile index*/, ckernel::DEFAULT_TENSOR_SHAPE);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_PACK
