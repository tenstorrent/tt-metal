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
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS);
            return;
        }
        else
        {
            for (std::uint32_t rc_i = (*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 0) & 0xFFu; rc_i != 0; --rc_i) // RC experiment: per-thread delay
            {
                asm volatile("nop");
            }
            // RC experiment (loop knock-out): byte 3 of L1 0x1FFE8 bit 1 -> unpack instructions wait until all packers are idle
            const bool rc_ko_unpack_after_pack = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x2u) != 0;
            // RC experiment (loopko): bits 4/5 (quiet math/pack RISC) also drop the per-iteration timeline markers
            const bool rc_ko_markers = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x30u) == 0;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if (rc_ko_markers)
                {
                    TIMESTAMP_DATA("RC_IT", loop) // RC experiment: timeline marker
                }
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        if (rc_ko_unpack_after_pack)
                        {
                            TTI_STALLWAIT(p_stall::STALL_UNPACK | p_stall::STALL_THCON, p_stall::PACK); // RC experiment
                        }
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
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
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
            for (std::uint32_t rc_i = (*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 8) & 0xFFu; rc_i != 0; --rc_i) // RC experiment: per-thread delay
            {
                asm volatile("nop");
            }
            // RC experiment (loop knock-out): byte 3 of L1 0x1FFE8 bit 2 -> the math RISC spins on the MATH_PACK semaphore
            // register (no L1 access) until pack has released DEST, so its next-block code fetches do not overlap packing.
            const bool rc_ko_math_spin = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x4u) != 0;
            // bit 3: same spin, but at the top of each iteration (before the timeline marker's L1 stores and code)
            const bool rc_ko_top_spin = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x8u) != 0;
            // RC experiment (loopko, knock-out a): bit 4 -> quiet math RISC. After each block it queues the next block's
            // DEST wait with an all-thread block mask plus RC_FILL math-type no-ops (INCRWC 0), so the RISC blocks on a full
            // instruction FIFO inside an icache-resident loop for the whole pack, with no L1 fetch, load or store.
            const bool rc_ko_quiet_math = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x10u) != 0;
            const bool rc_ko_markers    = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x30u) == 0;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if (rc_ko_top_spin)
                {
                    while (ckernel::semaphore_read(semaphore::MATH_PACK) != 0)
                    {
                    }
                }
                if (rc_ko_markers)
                {
                    TIMESTAMP_DATA("RC_IT", loop) // RC experiment: timeline marker
                }
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if (rc_ko_math_spin)
                    {
                        while (ckernel::semaphore_read(semaphore::MATH_PACK) != 0)
                        {
                        }
                    }
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    LLK_ASSERT(
                        (NUM_TILES_IN_BLOCK <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Matmul block exceeds destination capacity");
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, CT_DIM, RT_DIM);
                    }
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    if (rc_ko_quiet_math)
                    {
                        TTI_SEMWAIT(p_stall::STALL_THREAD, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX); // RC experiment
#pragma GCC unroll 1
                        for (std::uint32_t rc_f = 24; rc_f != 0; --rc_f)
                        {
                            TTI_INCRWC(0, 0, 0, 0); // RC experiment: FIFO filler, blocked behind the SEMWAIT
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_math_matmul_uninit_();
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
        // RC experiment (loopko, knock-out b): bits 6-7 of byte 3 = packer mode. 1: packer 0 alone packs the 4 faces
        // (4 PACRs, no other requester on the DEST read crossbar, no packer write on L1 ports 1-3). 2: packers 0+1,
        // packer 0 faces 0-1, packer 1 faces 2-3 (2 PACRs; no packer 2/3, no packer write on ports 2/3).
        const std::uint32_t rc_pack_mode = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 30) & 0x3u);
        if (rc_pack_mode == 1)
        {
            ckernel::ckernel_template tmp(1, 3, TT_OP_PACR(ADDR_MOD_0, 0, PACK_SEL(1), 0, 1, 0, 0), TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
            tmp.set_end_op(TT_OP_PACR(ADDR_MOD_1, 0, PACK_SEL(1), 0, 1, 0, 1));
            tmp.program();
        }
        else if (rc_pack_mode == 2)
        {
            // packer 1 reads DEST from face 2 (row 0x20) and writes L1 at face 2 (2 faces of Float32 = 128 16-B words;
            // the HW adds packer 0's header word, hence -1 as in set_packer_l1_offset). Float32 output only.
            TTI_SETDMAREG(0, 0x20, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 1));
            select_packer_dest_registers<dest_sync>();
            TT_SETDMAREG(0, 2 * 64 - 1, 0, LO_16(p_gpr_pack::TMP_LO));
            TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
            ckernel::ckernel_template tmp(1, 1, TT_OP_PACR(ADDR_MOD_0, 0, PACK_SEL(2), 0, 1, 0, 0), TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
            tmp.set_end_op(TT_OP_PACR(ADDR_MOD_1, 0, PACK_SEL(2), 0, 1, 0, 1));
            tmp.program();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
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
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t rc_i = (*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 16) & 0xFFu; rc_i != 0;
                 --rc_i) // RC experiment: per-thread delay
            {
                asm volatile("nop");
            }
            // RC experiment (loop knock-out): byte 3 of L1 0x1FFE8 bit 0 -> before every tile, wait until all packers are idle
            const bool rc_ko_pack_idle = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x1u) != 0;
            // RC experiment (loopko, knock-out a): bit 5 -> quiet pack RISC: no timeline marker, and the result address
            // comes from registers (base + tile * stride, read once here) instead of the runtime-args struct in L1.
            const bool rc_ko_quiet_pack       = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x20u) != 0;
            const bool rc_ko_markers          = ((*reinterpret_cast<volatile std::uint32_t*>(0x1FFE8u) >> 24) & 0x30u) == 0;
            const std::uint32_t rc_res_base   = L1_ADDRESS(buffer_Res[0]);
            const std::uint32_t rc_res_stride = L1_ADDRESS(buffer_Res[1]) - rc_res_base;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                if (rc_ko_markers)
                {
                    TIMESTAMP_DATA("RC_IT", loop) // RC experiment: timeline marker
                }
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    if (rc_ko_quiet_pack)
                    {
                        std::uint32_t rc_addr = rc_res_base + static_cast<std::uint32_t>(block) * NUM_TILES_IN_BLOCK * rc_res_stride;
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile, rc_addr += rc_res_stride)
                        {
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX + tile, rc_addr);
                        }
                    }
                    else
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            if (rc_ko_pack_idle)
                            {
                                TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON | p_stall::STALL_PACK, p_stall::PACK); // RC experiment
                            }
                            const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX + tile, L1_ADDRESS(buffer_Res[result_tile]));
                        }
                    }
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
