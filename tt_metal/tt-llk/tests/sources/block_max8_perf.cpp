// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK
#include "experimental/llk_unpack_block_max8_fpu.h"
#include "llk_unpack_common.h"

/** @brief Publish the -inf/1.0 sources for each FPU call (two zones of LOOP_FACTOR calls). */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    for (std::uint32_t i = 0; i < 2 * LOOP_FACTOR; ++i)
    {
        _llk_unpack_block_max8_fpu_();
    }
}
#endif

#ifdef LLK_TRISC_MATH
#include "experimental/llk_math_block_max8_fpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace
{

/** @brief Empty callback: measures the unary SFPU dispatcher alone. */
void dispatch_only(std::uint32_t)
{
}

/** @brief The reduction loop of _calculate_block_max8_ without masking or compaction. */
void reduce_only(std::uint32_t)
{
    for (std::uint32_t band = 0; band < TILE_R_DIM / 4; band += 2)
    {
        sfpu::block_max8::reduce_eight_rows(sfpu::block_max8::face_row_address(band));
    }
}

/** @brief The compaction loop of _calculate_block_max8_ without masking or reduction. */
void compact_only(std::uint32_t)
{
    for (std::uint32_t band = 0; band < TILE_R_DIM / 4; ++band)
    {
        const std::uint32_t address = sfpu::block_max8::face_row_address(band);
        const std::uint32_t packed  = (band / 4) * 4;
        sfpu::block_max8::compact_row_band<1>(address, packed + 2, band % 4);
        sfpu::block_max8::compact_row_band<0>(address, packed, band % 4);
    }
}

template <typename Callable>
void repeat(Callable&& fn, std::uint32_t loops, std::uint32_t valid_scores)
{
    for (std::uint32_t i = 0; i < loops; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(fn, 0 /*dst_index*/, VectorMode::RC_custom, valid_scores);
    }
}

} // namespace

/**
 * @brief Time each block_max8 phase on DST tile 0, LOOP_FACTOR calls per zone.
 *
 * Neither path's cost depends on DST contents (v_if is predication, not a branch;
 * the FPU path has no data-dependent control), so DST is left uninitialized and
 * pack stays idle. The FPU zones also wait on the unpacker's per-call sources.
 * Each zone ends with a Tensix sync so the last call's work is inside the measurement.
 */
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_block_max8_fpu_init_();
    constexpr std::uint32_t loops   = LOOP_FACTOR;
    constexpr std::uint32_t partial = 513; // any value below 1024 takes the masking path
    {
        ZONE_SCOPED("DISPATCH")
        repeat(dispatch_only, loops, 0 /*valid_scores*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("FULL_TILE")
        repeat(sfpu::_calculate_block_max8_, loops, TILE_R_DIM * TILE_C_DIM);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("PARTIAL_TILE")
        repeat(sfpu::_calculate_block_max8_, loops, partial);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("MASK")
        repeat(sfpu::_calculate_block_max8_mask_, loops, partial);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("REDUCE")
        repeat(reduce_only, loops, 0 /*valid_scores*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("COMPACT")
        repeat(compact_only, loops, 0 /*valid_scores*/);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("FPU_FULL_TILE")
        for (std::uint32_t i = 0; i < loops; ++i)
        {
            _llk_math_block_max8_fpu_(0 /*dst_index*/, TILE_R_DIM * TILE_C_DIM);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("FPU_PARTIAL_TILE")
        for (std::uint32_t i = 0; i < loops; ++i)
        {
            _llk_math_block_max8_fpu_(0 /*dst_index*/, partial);
        }
        PROFILER_SYNC();
    }
}
#endif

#ifdef LLK_TRISC_PACK
void run_kernel(RUNTIME_PARAMETERS params)
{
}
#endif
