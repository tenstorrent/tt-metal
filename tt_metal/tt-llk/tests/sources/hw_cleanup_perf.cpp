// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole compute_kernel_hw_cleanup: per iteration one identity datacopy of a tile, then with
// PERF_STAGE 1 the three per-thread cleanups, and the re-init each thread needs after one (PERF_STAGE 0 runs without).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync HW_DST_SYNC = ckernel::DstSync::SyncHalf;
constexpr std::uint32_t HW_NUM_FACES          = 4;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_hw_cleanup.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, HW_NUM_FACES, HW_NUM_FACES);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, HW_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
            if constexpr (PERF_STAGE >= 1)
            {
                _llk_unpack_hw_cleanup_canonical_<is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_hw_cleanup.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<HW_DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, ckernel::PackMode::Default>(
                HW_NUM_FACES, formats.math);
            _llk_math_wait_for_dest_available_<HW_DST_SYNC>();
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, HW_DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                0, formats.math, formats.math, HW_NUM_FACES);
            _llk_math_dest_section_done_<HW_DST_SYNC, is_fp32_dest_acc_en>();
            if constexpr (PERF_STAGE >= 1)
            {
                _llk_math_hw_cleanup_canonical_<HW_DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_hw_cleanup.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

inline void pack_configure_and_init(const FormatConfig& formats)
{
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * HW_NUM_FACES, FACE_R_DIM, ckernel::TILE_C_DIM, HW_NUM_FACES);
    _llk_pack_init_<ckernel::PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, ckernel::TILE_C_DIM, HW_NUM_FACES, 1, false);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        pack_configure_and_init(formats);
        _llk_pack_dest_init_<HW_DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            if (loop > 0)
            {
                pack_configure_and_init(formats);
            }
            _llk_packer_wait_for_math_done_();
            _llk_pack_<HW_DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
            _llk_pack_dest_section_done_<HW_DST_SYNC, is_fp32_dest_acc_en>();
            // The math side of the cleanup waits for MATH_PACK to drain, so the section is released first.
            if constexpr (PERF_STAGE >= 1)
            {
                _llk_pack_hw_cleanup_canonical_<HW_DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
