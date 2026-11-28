// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf driver for the Quasar EMA entry, laid out as ema_compute.cpp's EMA step: one DEST section per tile,
// transpose_tile into DEST tile 0 (unpack with transpose, datacopy), ema_tile(0), pack DEST tile 1. Each
// measurement pass is one chain: the carry is cleared once, then runs over all TILE_CNT tiles.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// ema_tile's input and output DEST tiles; the output offset mirrors sfpu::EMA_OUTPUT_TILE_DELTA.
constexpr std::uint32_t EMA_IN_TILE  = 0;
constexpr std::uint32_t EMA_OUT_TILE = EMA_IN_TILE + 1;

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

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
#endif
    // One tile per DEST section, as ema_compute acquires DEST per tile.
    const std::uint32_t num_sections        = TILE_CNT;
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    {
        START_PERF_MEASURE("INIT")
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_unpack = ckernel::trisc::BFD_ID_UNP0;

        // transpose_init: a 16-bit operand transposes in the unpacker.
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, tensor_shape, 1 /*num_tiles*/);

        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            // CFG wait masks persist across run types on the same device.
            // Isolated/congested UNPACK must not inherit the L1_TO_L1 chain.
            set_up_zero_dest_dvalid_handshake_for_unpack();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Quasar's 32-bit A2D datacopy is ELWADD and consumes SrcA
            // plus the dummy SrcB dvalid emitted by the unpack MOP.
            _perf_unpack_loop_set_valid<true /*set_a*/, is_fp32_dest_acc_en>(LOOP_FACTOR * TILE_CNT);
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t section = 0; section < num_sections; ++section)
                {
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(section /*l1_tile_idx*/, tensor_shape);
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
    // No uninit: the production op has none on Quasar.
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

// The entry reads the compute kernel's DST_SYNC_MODE / DST_ACCUM_MODE.
constexpr ckernel::DstSync DST_SYNC_MODE = dest_sync;
constexpr bool DST_ACCUM_MODE            = is_fp32_dest_acc_en;

#include "llk_sfpu/llk_math_ema_sfpu_entry.h"

using namespace ckernel;
using namespace ckernel::math;

static_assert(EMA_OUT_TILE - EMA_IN_TILE == sfpu::EMA_OUTPUT_TILE_DELTA, "the packed tile must be the entry's output");

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

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
#endif
    const std::uint32_t num_sections = TILE_CNT;
    const DataFormat src_format      = static_cast<DataFormat>(formats.math);

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format);
    {
        START_PERF_MEASURE("INIT")

        // ema_init, then transpose_init's datacopy init. EMA's body sits in replay bank 1, apart from the datacopy MOP.
        llk_math_ema_sfpu_init();
        llk_math_ema_sfpu_load_alpha_beta(EMA_ALPHA_BITS, EMA_BETA_BITS);
        _configure_alu_cfg_<AluCfgType::DEFAULT, is_fp32_dest_acc_en>(src_format, src_format);
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM, 1 /*num_matrices*/);

        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            // CFG wait masks persist across run types. MATH_ISOLATE runs without
            // a pack consumer; UNPACK_ISOLATE only mocks Src
            // handshakes. Neither may inherit the FPU→SFPU→PACK chain.
            set_up_zero_dest_dvalid_handshake_for_math();
            set_up_zero_dest_dvalid_handshake_for_sfpu();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, is_fp32_dest_acc_en>(LOOP_FACTOR * TILE_CNT);
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                llk_math_ema_sfpu_clear_previous_output();
                for (std::uint32_t section = 0; section < num_sections; ++section)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_();
                    }
                    _llk_math_eltwise_unary_datacopy_(EMA_IN_TILE);
                    llk_math_ema_sfpu_tile(EMA_IN_TILE);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
                    {
                        // Stands in for the idle pack's release, so MATH_PACK never fills up.
                        ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
                    }
                }
            }
            // Drain every execution unit driven by T1 before PACK takes over.
            wait_sfpu_idle();
            wait_fpu_idle();
            wait_mop_idle();
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

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
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    const std::uint32_t num_sections = TILE_CNT;

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    }
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(
        ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*pack_tile: the output tile only*/);
    {
        START_PERF_MEASURE("INIT")

        if constexpr (PERF_RUN_TYPE != PerfRunType::L1_TO_L1)
        {
            // PACK_ISOLATE packs independently. UNPACK_ISOLATE /
            // MATH_ISOLATE do not pack. None may inherit the L1_TO_L1 wait mask.
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t section = 0; section < num_sections; ++section)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    _llk_pack_(EMA_OUT_TILE, section /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
                    {
                        // Stands in for the idle math's post, so the section done's SEMGET never underflows.
                        ckernel::trisc::t6_semaphore_post(ckernel::trisc::semaphore::MATH_PACK);
                    }
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif
