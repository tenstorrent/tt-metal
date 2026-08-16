// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Cost of a compression-enabled PACR on Blackhole.
//
// sources/pack_zero_compress_test.cpp proved (bit-exact decode) that the packer can
// do threshold + compaction with zero SFPU instructions. It never priced it. This
// kernel does: it packs the SAME Dest tile PACK_ITER_COUNT times inside the TILE_LOOP
// zone with the compression path either disabled (baseline) or enabled, so a two-point
// slope over PACK_ITER_COUNT yields the marginal cost of one tile pack.
//
// PACK_ISOLATE is the run type that matters here: the work is on the pack thread.
// UNPACK_ISOLATE reuses the same kernel and the same tile to price the unpacker,
// because the steady-state rate of an SFPU-free filter pipeline is max(unpack, pack),
// not their sum.
//
// Compile-time knobs (emitted into build.h by the python driver):
//   COMPRESS_EN             - clear THCON_SEC0_REG1_Disable_zero_compress before packing
//   ROW_START_SECTION_SIZE  - THCON_SEC0_REG1_Row_start_section_size, in 16B units
//   DOWNSAMPLE_MASK         - THCON_SEC0_REG1_Downsample_mask (0 == disabled; ALWAYS
//                             written: set_packer_config never touches THCON_SEC0_REG1
//                             word 3, so a mask left behind by an earlier kernel
//                             survives an ELF reload and silently decimates this pack)
//   PACK_ITER_COUNT         - tile packs issued inside the timed zone
//   CYCLE_OUTPUT            - rotate the L1 destination over 16 tile slots instead of
//                             hammering one address
//
// RELU_CONFIG (runtime) carries the packer ReLU mode + threshold, so the same kernel
// prices plain pack, compress-only, relu-only and relu+compress.

#include <algorithm>
#include <cstdint>

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

static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces = params.num_faces;
    const auto& buffer_A          = params.buffer_A;
#endif
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);

        // One unpack outside the timed zone so Dest holds the sparse pattern the pack
        // arm is supposed to compress. PACK_ISOLATE would otherwise pack garbage and
        // the density sweep would be meaningless.
        _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t i = 0; i < PACK_ITER_COUNT; ++i)
            {
                _llk_unpack_A_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces = params.num_faces;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
            num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
            0, formats.math, formats.math, num_faces);
        _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // Math does only the hardware handshake the unpacker needs; it must not add
            // work of its own to an unpack-isolated measurement.
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t i = 0; i < PACK_ITER_COUNT; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                        0, formats.math, formats.math, num_faces);
                }
            }
            else
            {
                _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(PACK_ITER_COUNT * TILE_NUM_FACES);
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t num_faces = params.num_faces;
    const int RELU_CONFIG         = params.RELU_CONFIG;
    const auto& buffer_Res        = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src,
            formats.pack_dst,
            FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES /* tile_size */,
            FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            false /* partial_face */,
            false /* narrow_tile */,
            RELU_CONFIG);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        // set_packer_config forces config.f.uncompress = 1, so compression has to be
        // re-enabled behind the LLK's back. Row_start_section_size reserves room for the
        // row-start-index array that the compressed layout puts before the data.
        {
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
            if constexpr (ROW_START_SECTION_SIZE != 0)
            {
                cfg_reg_rmw_tensix<THCON_SEC0_REG1_Row_start_section_size_RMW>(ROW_START_SECTION_SIZE);
            }
            if constexpr (COMPRESS_EN)
            {
                cfg_reg_rmw_tensix<THCON_SEC0_REG1_Disable_zero_compress_RMW>(0);
            }
            // Written unconditionally, including the disabled (0) case -- see the header
            // comment on DOWNSAMPLE_MASK.
            cfg_reg_rmw_tensix<THCON_SEC0_REG1_Downsample_mask_RMW>(DOWNSAMPLE_MASK);
        }

        _llk_packer_wait_for_math_done_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t i = 0; i < PACK_ITER_COUNT; ++i)
            {
                // Rotating over 16 tile slots keeps the write stream moving through L1
                // the way a real cascade pass would, instead of re-writing one line.
                const std::uint32_t byte_addr = CYCLE_OUTPUT ? (PERF_OUTPUT + ((i & 0xF) << 12)) : PERF_OUTPUT;
                _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default>(0, byte_addr / 16 - 1);
            }
            // The RISC runs far ahead of the packer; without this the zone-end timestamp
            // could land before the last PACRs have retired.
            TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
        }
        PROFILER_SYNC();
    }
    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

    // Outside every timed zone: report how many bytes the last pack actually emitted.
    // Without this a build where the compression config write silently failed would
    // measure identically to the baseline and read as "compression is free".
    if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
    {
        TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
        tensix_sync();
        volatile std::uint32_t* diag = reinterpret_cast<volatile std::uint32_t*>(buffer_Res[0]);
        diag[0]                      = 0xC0DEBA5E;
        diag[1]                      = reg_read(RISCV_TDMA_REG_PACKED_SIZE + 0x080); // PackerTileSize(0, T2), 16B units
        diag[2]                      = COMPRESS_EN ? 1u : 0u;
        diag[3]                      = PACK_ITER_COUNT;
    }
}

#endif // LLK_TRISC_PACK
