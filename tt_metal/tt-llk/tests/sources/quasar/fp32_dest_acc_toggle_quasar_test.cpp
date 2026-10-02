// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Mid-kernel FP32 dest-acc toggle (_llk_set_fp32_dest_acc_) on Quasar.
//
// The kernel is compiled for 16-bit dest. It runs the same accumulation twice into dest tile 0:
//   phase 0 (16-bit dest):  ELWADD acc_to_dest over TILES_PER_PHASE input tiles, pack as Float16_b
//   toggle to 32-bit dest on all three threads
//   phase 1 (32-bit dest):  the same accumulation, pack from Float32 dest to Float16_b
// The stimulus is chosen so that the two phases give different, rounding-mode-independent results: the
// addends are below half a bf16 ULP of the first tile, so a 16-bit dest drops every one of them, while a
// 32-bit dest keeps them and their sum is exactly one bf16 ULP.
//
// Everything width-dependent is given the phase's width explicitly (math dest-section helpers, pack
// hw_configure / section done, pack IN_DATA_FORMAT). Only dest tile 0 is used and only DstSync::SyncFull
// is supported, so the dest bank geometry change between widths does not come into play.

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_fp32_dest_acc.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Input tiles accumulated into one dest tile per phase; must match TILES_PER_PHASE in the python test.
constexpr std::uint32_t TILES_PER_PHASE = 9;
constexpr std::uint32_t NUM_PHASES      = 2;

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_binary_operands.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    static_assert(dest_sync == DstSync::SyncFull, "fp32 dest-acc toggle test only supports DstSync::SyncFull");
    static_assert(!is_fp32_dest_acc_en, "fp32 dest-acc toggle test must be compiled for 16-bit dest");
    static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "fp32 dest-acc toggle test is functional only");
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
    const Operand& buffer_B = params.buffer_B;
#endif

    {
        ZONE_SCOPED("INIT")
        set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        // Float16_b src registers are valid for both 16-bit and 32-bit dest, so the unpacker needs no
        // reconfiguration across the toggle.
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
        _llk_unpack_binary_operands_init_(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            1 /*num_tiles_per_unpack*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t phase = 0; phase < NUM_PHASES; ++phase)
        {
            if (phase == 1)
            {
                _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>();
            }
            for (std::uint32_t i = 0; i < TILES_PER_PHASE; ++i)
            {
                const std::uint32_t tile = phase * TILES_PER_PHASE + i;
                _llk_unpack_binary_operands_(tile, tile);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"
#include "tensor_shape.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    static_assert(dest_sync == DstSync::SyncFull, "fp32 dest-acc toggle test only supports DstSync::SyncFull");
    static_assert(!is_fp32_dest_acc_en, "fp32 dest-acc toggle test must be compiled for 16-bit dest");
    static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "fp32 dest-acc toggle test is functional only");
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    {
        ZONE_SCOPED("INIT")
        set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();

        const DataFormat math_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, ckernel::MathFidelity::LoFi>(ckernel::DEFAULT_TENSOR_SHAPE, true /*acc_to_dest*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t phase = 0; phase < NUM_PHASES; ++phase)
        {
            if (phase == 1)
            {
                // Flips ALU_ACC_CTRL Fp32_enabled / SFPU_Fp32_enabled and invalidates the ALU format latch.
                // The ELWADD MOP and the DEFAULT-set src formats are width-independent, so no re-init is
                // needed: this checks that the two bits alone switch the FPU to 32-bit dest.
                _llk_math_set_fp32_dest_acc_(true);
            }
            for (std::uint32_t i = 0; i < TILES_PER_PHASE; ++i)
            {
                _llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD>(0 /*dest tile*/, ckernel::DEFAULT_TENSOR_SHAPE);
            }
            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    static_assert(dest_sync == DstSync::SyncFull, "fp32 dest-acc toggle test only supports DstSync::SyncFull");
    static_assert(!is_fp32_dest_acc_en, "fp32 dest-acc toggle test must be compiled for 16-bit dest");
    static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "fp32 dest-acc toggle test is functional only");
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
#endif

    {
        ZONE_SCOPED("INIT")
        set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
            ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, false /*EN_32BIT_DEST*/>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        // Phase 0: 16-bit dest.
        _llk_pack_(0 /*dest tile*/, 0 /*l1 tile*/, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_pack_dest_dvalid_section_done_<dest_sync, false /*EN_32BIT_DEST*/>();

        // Toggle. Quasar has no Read_32b_data bit: the packer reads 32-bit dest because IN_DATA_FORMAT
        // is Float32, so reprogram it explicitly (the host table only holds the compiled 16-bit mode).
        _llk_set_fp32_dest_acc_<ThreadId::PackThreadId>();
        _llk_pack_reconfig_data_format_<p_pacr::PACK0>(to_underlying(DataFormat::Float32), formats.pack_dst);

        // Phase 1: 32-bit dest.
        _llk_pack_(0 /*dest tile*/, 1 /*l1 tile*/, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_pack_dest_dvalid_section_done_<dest_sync, true /*EN_32BIT_DEST*/>();
        PROFILER_SYNC();
    }
}

#endif
