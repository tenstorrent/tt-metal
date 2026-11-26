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
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// A run of TILE_CNT tiles is TILE_CNT / tiles_per_block operand sets, each staged into its own Dest section.
inline std::uint32_t binary_sfpu_tiles_per_block(const std::uint32_t src0_tile_idx, const std::uint32_t src1_tile_idx, const std::uint32_t dst_tile_idx)
{
    const std::uint32_t max_operand = src0_tile_idx > src1_tile_idx ? src0_tile_idx : src1_tile_idx;
    return (max_operand > dst_tile_idx ? max_operand : dst_tile_idx) + 1;
}

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_dest_borrow.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

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
    const Operand& buffer_A             = params.buffer_A;
    const std::uint32_t SRC0_TILE_IDX   = params.SRC0_TILE_IDX;
    const std::uint32_t SRC1_TILE_IDX   = params.SRC1_TILE_IDX;
    const std::uint32_t DST_TILE_IDX    = params.DST_TILE_IDX;
#endif
    const std::uint32_t tiles_per_block = binary_sfpu_tiles_per_block(SRC0_TILE_IDX, SRC1_TILE_IDX, DST_TILE_IDX);
    const std::uint32_t num_blocks      = dest_section_count(TILE_CNT, tiles_per_block);
    const std::uint32_t num_passes      = LOOP_FACTOR * num_blocks;

    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1 && unpack_to_dest)
        {
            // Isolated/congested UNP_DEST execution must not inherit a dest-dvalid wait mask.
            set_up_zero_dest_dvalid_handshake_for_unpack();
        }

        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(
            ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_unpack = ckernel::trisc::BFD_ID_UNP0;

        if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
        {
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        }

        if constexpr (unpack_to_dest)
        {
            // The to-dest placers carry no synchronization: in L1_TO_L1 a Dest loan from math brackets them and
            // sets the section base, so every section starts at DEST tile 0.
            _llk_unpack_dest_init_();
            _llk_unpack_unary_operand_to_dest_init_(bfd_unpack);
        }
        else
        {
            _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
                bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, tiles_per_block);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                // Quasar's 32-bit A2D datacopy is ELWADD and consumes SrcA
                // plus the dummy SrcB dvalid emitted by the unpack MOP.
                _perf_unpack_loop_set_valid<true /*set_a*/, is_fp32_dest_acc_en>(num_passes * tiles_per_block);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t pass = 0; pass < num_passes; ++pass)
            {
                const std::uint32_t l1_tile_idx = (pass % num_blocks) * tiles_per_block;
                if constexpr (unpack_to_dest && (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE))
                {
                    _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                }
                if constexpr (unpack_to_dest)
                {
                    _llk_unpack_unary_operand_to_dest_block_(l1_tile_idx, 0 /*dst_tile_idx*/, tiles_per_block);
                }
                else
                {
                    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(l1_tile_idx, ckernel::DEFAULT_TENSOR_SHAPE);
                }
                if constexpr (unpack_to_dest && (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE))
                {
                    _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
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
#include "llk_dest_borrow.h"
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
    const std::uint32_t ZERO_POINT      = params.ZERO_POINT;
#endif
    const std::uint32_t tiles_per_block = binary_sfpu_tiles_per_block(SRC0_TILE_IDX, SRC1_TILE_IDX, DST_TILE_IDX);
    const std::uint32_t num_passes      = LOOP_FACTOR * (TILE_CNT / tiles_per_block);
    // Each block stages both operands from buffer_A into Dest, as laid out by the test.
    LLK_ASSERT(tiles_per_block == 2 || tiles_per_block == 4, "Binary SFPU expects 2 or 4 staged tiles per block");
    LLK_ASSERT(TILE_CNT % tiles_per_block == 0, "Binary SFPU expects whole blocks of staged tiles");
    const DataFormat math_format = static_cast<DataFormat>(formats.math);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE && unpack_to_dest))
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // FPU datacopy (when present) and SFPU execute sequentially on this
            // thread. Keep them on one Dest bank instead of creating a dvalid
            // chain whose producer would also wait for inactive UNPACK/PACK.
            set_up_zero_dest_dvalid_handshake_for_math();
            set_up_zero_dest_dvalid_handshake_for_sfpu();
        }

        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);

        if constexpr (!unpack_to_dest)
        {
            _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1 /*num_matrices*/);
        }
        // SFPU uses ADDR_MOD_7 and does not overwrite the datacopy MOP's
        // ADDR_MOD_0/1 or bank-0 programming, so both initializers can remain in
        // the INIT zone before the measured TILE_LOOP.
        _llk_math_eltwise_sfpu_init_();
        test_utils::init_binary_sfpu_operation_quasar<SFPU_BINARY_OP, is_fp32_dest_acc_en, SFPU_SIGN_MAGNITUDE, APPROX_MODE, BROADCAST_TYPE>(ZERO_POINT);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, is_fp32_dest_acc_en>(num_passes * tiles_per_block);
            }
            else
            {
                // Stand-in lender: unpack takes each section's DEST bank through the loan, as in production.
                for (std::uint32_t pass = 0; pass < num_passes; ++pass)
                {
                    _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    // Pack is idle, so math takes MATH_PACK back.
                    ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t pass = 0; pass < num_passes; ++pass)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_();
                }
                if constexpr (!unpack_to_dest)
                {
                    for (std::uint32_t i = 0; i < tiles_per_block; ++i)
                    {
                        _llk_math_eltwise_unary_datacopy_(DST_INDEX + i);
                    }
                }
                else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                    _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                }
                test_utils::call_binary_sfpu_operation_quasar<
                    SFPU_BINARY_OP,
                    dest_sync,
                    is_fp32_dest_acc_en,
                    SFPU_DST_ROUNDING_MODE,
                    SFPU_ITERATIONS,
                    SFPU_SIGN_MAGNITUDE,
                    APPROX_MODE,
                    BROADCAST_TYPE>(SRC0_TILE_IDX, SRC1_TILE_IDX, DST_TILE_IDX, math_format);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t DST_TILE_IDX    = params.DST_TILE_IDX;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_Res           = params.buffer_Res;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t SRC0_TILE_IDX   = params.SRC0_TILE_IDX;
    const std::uint32_t SRC1_TILE_IDX   = params.SRC1_TILE_IDX;
#endif
    const std::uint32_t num_blocks = dest_section_count(TILE_CNT, binary_sfpu_tiles_per_block(SRC0_TILE_IDX, SRC1_TILE_IDX, DST_TILE_IDX));

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
    {
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    }
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // PACK runs without the L1_TO_L1 producer chain in isolated modes.
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(
            ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;

        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*only the SFPU result tile*/);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t pass = 0; pass < LOOP_FACTOR * num_blocks; ++pass)
            {
                // One result tile per block.
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                _llk_pack_(DST_TILE_IDX, pass % num_blocks /*tile index*/, ckernel::DEFAULT_TENSOR_SHAPE);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_PACK
