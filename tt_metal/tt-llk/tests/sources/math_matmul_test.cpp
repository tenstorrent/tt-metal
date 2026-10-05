// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const std::uint32_t in1_tile_r_dim = params.in1_tile_r_dim;

    const bool PARTIAL_FACE_A = params.PARTIAL_FACE_A;
    const bool PARTIAL_FACE_B = params.PARTIAL_FACE_B;

    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const int NUM_BLOCKS              = params.NUM_BLOCKS;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const Operand& buffer_A           = params.buffer_A;
    const Operand& buffer_B           = params.buffer_B;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B, // in1
            num_faces_A, // in0
            TILE_SIZE_UNPACK_B,
            TILE_SIZE_UNPACK_A);
        _llk_unpack_AB_matmul_init_<>(
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM,
            KT_DIM,
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B,     // in1
            num_faces_A,     // in0
            PARTIAL_FACE_B,  // in1
            PARTIAL_FACE_A); // in0
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        // Experiment: REPRO_UDELAY spins at the start of this thread's TILE_LOOP zone (constant code size).
        {
            std::uint32_t repro_uspin;
            asm volatile(".option push\n\t.option norvc\n\tli %0, %1\n2:\n\taddi %0, %0, -1\n\tbnez %0, 2b\n\t.option pop" : "=&r"(repro_uspin) : "i"(REPRO_UDELAY + 1));
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // Experiment (bistability repro): one real unpack (an L1 read by the unpacker) at spin REPRO_UNP_AT.
            if constexpr (REPRO_UNP_AT != 0)
            {
                for (std::uint32_t repro_k = 0; repro_k < static_cast<std::uint32_t>(REPRO_UNP_AT); repro_k++)
                {
                    asm volatile("nop");
                }
                _llk_unpack_AB_matmul_<>(L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]), 0, 0, TILE_SIZE_UNPACK_B, TILE_SIZE_UNPACK_A,
                    PARTIAL_FACE_B, PARTIAL_FACE_A, CT_DIM, RT_DIM, KT_DIM);
            }
            // Experiment (bistability repro): idle before this thread ends its zone.
            for (std::uint32_t repro_i = 0; repro_i < static_cast<std::uint32_t>(REPRO_TAIL + REPRO_TAIL_U); repro_i++)
            {
                asm volatile("nop");
            }
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_unpack_AB_matmul_<>(
                            L1_ADDRESS(buffer_A[0]),
                            L1_ADDRESS(buffer_B[0]),
                            j,
                            j * CT_DIM,
                            TILE_SIZE_UNPACK_B,
                            TILE_SIZE_UNPACK_A,
                            PARTIAL_FACE_B,
                            PARTIAL_FACE_A,
                            CT_DIM,
                            RT_DIM,
                            KT_DIM);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH
static std::uint32_t repro_rt_at = 0; // experiment: runtime injection iteration

#include "llk_math_common.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const std::uint32_t in0_tile_c_dim = params.in0_tile_c_dim;
    const std::uint32_t in1_tile_r_dim = params.in1_tile_r_dim;
    const std::uint32_t in1_tile_c_dim = params.in1_tile_c_dim;

    const bool PARTIAL_FACE_MATH = params.PARTIAL_FACE_MATH;

    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int DST_INDEX                    = params.DST_INDEX;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const int NUM_BLOCKS              = params.NUM_BLOCKS;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
            in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, PARTIAL_FACE_MATH, UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM);
        repro_rt_at = *reinterpret_cast<volatile std::uint32_t*>(0x16AFE4); // experiment: host-written, read before TILE_LOOP
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        // Experiment: REPRO_UDELAY spins at the start of this thread's TILE_LOOP zone (constant code size).
        {
            std::uint32_t repro_uspin;
            asm volatile(".option push\n\t.option norvc\n\tli %0, %1\n2:\n\taddi %0, %0, -1\n\tbnez %0, 2b\n\t.option pop" : "=&r"(repro_uspin) : "i"(REPRO_UDELAY + 1));
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // Experiment (bistability repro): one controlled operation REPRO_INJ_AT spins after the zone opens.
            if constexpr (REPRO_INJ == 15)
            {
                // Same code for every value: one L1 load at the runtime spin iteration read in INIT (0 = none).
                volatile std::uint32_t* repro_l1 = &llk_profiler::buffer[llk_profiler::TRISC_ID][llk_profiler::BUFFER_LENGTH - 1];
                std::uint32_t repro_n, repro_v;
                std::uint32_t repro_k = repro_rt_at == 0 ? 0xFFFFFFFFu : 20000 - repro_rt_at;
                asm volatile(
                    ".option push\n\t.option norvc\n\t"
                    "li %0, 20000\n"
                    "1:\n\tnop\n\tbne %0, %2, 2f\n\tlw %1, 0(%3)\n"
                    "2:\n\taddi %0, %0, -1\n\tbnez %0, 1b\n\t.option pop"
                    : "=&r"(repro_n), "=&r"(repro_v)
                    : "r"(repro_k), "r"(repro_l1)
                    : "memory");
            }
            else if constexpr (REPRO_INJ >= 11)
            {
                // One loop, one fixed code path: at iteration REPRO_INJ_AT the body falls through into a single
                // 4-byte slot: nop (11), L1 load (12), L1 store (13) or wall-clock MMIO read (14).
                volatile std::uint32_t* repro_l1 = &llk_profiler::buffer[llk_profiler::TRISC_ID][llk_profiler::BUFFER_LENGTH - 1];
                volatile std::uint32_t* repro_mmio = reinterpret_cast<volatile std::uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
                volatile std::uint32_t* repro_addr = (REPRO_INJ == 14) ? repro_mmio : repro_l1;
                std::uint32_t repro_n, repro_k, repro_v;
                asm volatile(
                    ".option push\n\t.option norvc\n\t"
                    "li %0, 20000\n\tli %1, %4\n"
                    "1:\n\tnop\n\tbne %0, %1, 2f\n\t"
                    ".if %5 == 12 || %5 == 14\n\tlw %2, 0(%3)\n\t.elseif %5 == 13\n\tsw zero, 0(%3)\n\t.else\n\tnop\n\t.endif\n"
                    "2:\n\taddi %0, %0, -1\n\tbnez %0, 1b\n\t.option pop"
                    : "=&r"(repro_n), "=&r"(repro_k), "=&r"(repro_v)
                    : "r"(repro_addr), "i"(20000 - REPRO_INJ_AT), "i"(REPRO_INJ)
                    : "memory");
            }
            else if constexpr (REPRO_INJ != 0)
            {
                for (std::uint32_t repro_k = 0; repro_k < static_cast<std::uint32_t>(REPRO_INJ_AT); repro_k++)
                {
                    asm volatile("nop");
                }
                if constexpr (REPRO_INJ == 1)
                {
                    llk_profiler::buffer[llk_profiler::TRISC_ID][llk_profiler::BUFFER_LENGTH - 1] = 0;
                    asm volatile("" ::: "memory");
                }
                else if constexpr (REPRO_INJ == 2)
                {
                    (void)ckernel::read_wall_clock();
                }
                else if constexpr (REPRO_INJ == 3)
                {
                    TTI_NOP;
                }
                else if constexpr (REPRO_INJ == 5)
                {
                    asm volatile(".option push\n\t.option norvc\n\t.rept 64\n\tnop\n\t.endr\n\t.option pop");
                }
                else if constexpr (REPRO_INJ == 6)
                {
                    (void)*reinterpret_cast<volatile std::uint32_t*>(&llk_profiler::buffer[llk_profiler::TRISC_ID][llk_profiler::BUFFER_LENGTH - 1]);
                }
                else if constexpr (REPRO_INJ == 7)
                {
                    asm volatile(".option push\n\t.option norvc\n\t.rept 8\n\tnop\n\t.endr\n\t.option pop");
                }
            }
            // Experiment (bistability repro): idle before this thread ends its zone.
            for (std::uint32_t repro_i = 0; repro_i < static_cast<std::uint32_t>(REPRO_TAIL + REPRO_TAIL_M); repro_i++)
            {
                asm volatile("nop");
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, CT_DIM, RT_DIM);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    LLK_ASSERT(
                        (NUM_TILES_IN_BLOCK <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Matmul block exceeds destination capacity");
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, CT_DIM, RT_DIM);
                    }
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_math_matmul_uninit_();
    if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        // Experiment (bistability repro): idle before this thread ends its zone.
        for (std::uint32_t repro_i = 0; repro_i < static_cast<std::uint32_t>(REPRO_TAIL_MA); repro_i++)
        {
            asm volatile("nop");
        }
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

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim     = params.in0_tile_r_dim;
    const bool PARTIAL_FACE_PACK           = params.PARTIAL_FACE_PACK;
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK     = params.TILE_SIZE_PACK;
    const int num_faces                    = params.num_faces;
    const int DST_INDEX                    = params.DST_INDEX;
    const std::uint32_t CT_DIM             = params.CT_DIM;
    const std::uint32_t RT_DIM             = params.RT_DIM;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src,
            formats.pack_dst,
            TILE_SIZE_PACK,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            PARTIAL_FACE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        // Experiment (bistability repro): REPRO_PAD skipped nops move the pack loop by 4*REPRO_PAD
        // bytes at a constant executed cost; REPRO_DELAY spins with constant code size.
        asm volatile(".option push\n\t.option norvc\n\tj 1f\n\t.rept %0\n\tnop\n\t.endr\n1:\n\t.option pop" ::"i"(REPRO_PAD));
        {
            std::uint32_t repro_spin;
            asm volatile(".option push\n\t.option norvc\n\tli %0, %1\n2:\n\taddi %0, %0, -1\n\tbnez %0, 2b\n\t.option pop" : "=&r"(repro_spin) : "i"(REPRO_DELAY + 1));
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                    {
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                        // Experiment (bistability repro): REPRO_RESET 1 = wait for the packers after every tile,
                        // 2 = after every 16th tile, 3 = once, before the first tile.
                        if constexpr (REPRO_RESET == 1)
                        {
                            TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::PACK);
                        }
                        else if constexpr (REPRO_RESET == 2)
                        {
                            if ((loop & 15) == 15)
                            {
                                TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::PACK);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX + tile, L1_ADDRESS(buffer_Res[result_tile]));
                    }
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    if constexpr (REPRO_RESET == 3) // experiment: drain the packers at every dest section
                    {
                        TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::PACK);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif