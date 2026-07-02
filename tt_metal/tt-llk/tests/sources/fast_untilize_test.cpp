// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BH fast-untilize LLK test (tt-metal#42048 + #42049).
// Fast-untilize unpack loads 2/3/4-tile chunks, custom math lays each chunk out for
// 4-interface row-major readout, and fast_untilize pack writes the RM strip.
//
// Current shapes: row-decomposed unit_dim={4,2,3}; compressed BFP inputs unpack
// one tile at a time to skip exponent sections. ct=1 is left to the integrated
// standard fallback path. Hardcoded constraint: num_faces=4.

#include <cstdint>

#include "ckernel.h"
#include "experimental/llk_fast_untilize_common.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr bool FAST_UNTILIZE_SINGLE_UNIT             = FULL_CT_DIM <= ckernel::FAST_UNTILIZE_MAX_UNIT_DIM;
constexpr std::uint32_t MAX_UNITS_PER_ROW            = (FULL_CT_DIM + ckernel::FAST_UNTILIZE_MAX_UNIT_DIM - 1) / ckernel::FAST_UNTILIZE_MAX_UNIT_DIM;
constexpr std::uint32_t FAST_UNTILIZE_FIRST_UNIT_DIM = ckernel::fast_untilize_next_unit_dim(FULL_CT_DIM);
constexpr bool FAST_UNTILIZE_BFP_B_INPUT =
    UNPACK_A_IN == ckernel::to_underlying(DataFormat::Bfp8_b) || UNPACK_A_IN == ckernel::to_underlying(DataFormat::Bfp4_b);
// Mirror production fast_untilize: test both ambient dest_sync values while
// running the private fast region with half-sync double buffering.
constexpr auto FAST_UNTILIZE_INTERNAL_DEST_SYNC = ckernel::FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE;
#ifndef DST_ACCUM_MODE
static constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;
#endif

static_assert(PERF_RUN_TYPE != PerfRunType::L1_CONGESTION, "L1 congestion mode is not supported for fast_untilize");
static_assert(BLOCK_CT_DIM == FULL_CT_DIM, "fast_untilize_test expects one full tile row per kernel instance");
static_assert(FULL_CT_DIM >= 2, "fast_untilize_test supports ct>=2; ct=1 uses the standard fallback path");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_fast_untilize.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
#endif

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        ckernel::_llk_unpack_fast_untilize_init_<DST_ACCUM_MODE>(
            formats.unpack_A_src, formats.unpack_A_dst, FAST_UNTILIZE_BFP_B_INPUT ? 1 : FAST_UNTILIZE_FIRST_UNIT_DIM);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true, is_fp32_dest_acc_en>(LOOP_FACTOR * FULL_RT_DIM * FULL_CT_DIM * 4);
            PROFILER_SYNC();
            return;
        }

        if constexpr (FAST_UNTILIZE_SINGLE_UNIT)
        {
            constexpr std::uint32_t unit_dim = FULL_CT_DIM;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    if constexpr (FAST_UNTILIZE_BFP_B_INPUT)
                    {
                        const std::uint32_t address         = L1_ADDRESS(buffer_A[rt * FULL_CT_DIM]);
                        const std::uint32_t tile_stride_16B = L1_ADDRESS(buffer_A[rt * FULL_CT_DIM + 1]) - address;
                        ckernel::_llk_unpack_fast_untilize_bfp_block_(address, tile_stride_16B, unit_dim);
                    }
                    else
                    {
                        ckernel::_llk_unpack_fast_untilize_block_(L1_ADDRESS(buffer_A[rt * FULL_CT_DIM]), unit_dim);
                    }
                }
            }
        }
        else
        {
            std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
            const std::uint32_t units_per_row = ckernel::fast_untilize_decompose_row(FULL_CT_DIM, unit_dims);
            std::uint32_t prev_unit_dim       = unit_dims[0];
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    std::uint32_t chunk_col = 0;
                    for (std::uint32_t u = 0; u < units_per_row; u++)
                    {
                        const std::uint32_t unit_dim = unit_dims[u];
                        if constexpr (FAST_UNTILIZE_BFP_B_INPUT)
                        {
                            const std::uint32_t address         = L1_ADDRESS(buffer_A[rt * FULL_CT_DIM + chunk_col]);
                            const std::uint32_t tile_stride_16B = L1_ADDRESS(buffer_A[rt * FULL_CT_DIM + chunk_col + 1]) - address;
                            ckernel::_llk_unpack_fast_untilize_bfp_block_(address, tile_stride_16B, unit_dim);
                        }
                        else
                        {
                            if (unit_dim != prev_unit_dim)
                            {
                                ckernel::_llk_unpack_fast_untilize_reinit_unit_dim_<DST_ACCUM_MODE>(unit_dim);
                                prev_unit_dim = unit_dim;
                            }
                            ckernel::_llk_unpack_fast_untilize_block_(L1_ADDRESS(buffer_A[rt * FULL_CT_DIM + chunk_col]), unit_dim);
                        }
                        chunk_col += unit_dim;
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("UNINIT")
        ckernel::_llk_unpack_fast_untilize_uninit_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_fast_untilize_api.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<FAST_UNTILIZE_INTERNAL_DEST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        llk_math_fast_untilize_init();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_loop_clear_valid<true, is_fp32_dest_acc_en>(LOOP_FACTOR * FULL_RT_DIM * FULL_CT_DIM * 4);
            PROFILER_SYNC();
            return;
        }

        if constexpr (FAST_UNTILIZE_SINGLE_UNIT)
        {
            constexpr std::uint32_t unit_dim = FULL_CT_DIM;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DEST_SYNC>();
                    }
                    llk_math_fast_untilize_block(0, unit_dim);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DEST_SYNC, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        else
        {
            std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
            const std::uint32_t units_per_row = ckernel::fast_untilize_decompose_row(FULL_CT_DIM, unit_dims);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    for (std::uint32_t u = 0; u < units_per_row; u++)
                    {
                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                        {
                            _llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DEST_SYNC>();
                        }
                        llk_math_fast_untilize_block(0, unit_dims[u]);
                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                        {
                            _llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DEST_SYNC, is_fp32_dest_acc_en>();
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }

    {
        // Keep this zone on a distinct line to avoid 16-bit profiler hash collisions.
        ZONE_SCOPED("UNINIT")
        llk_math_fast_untilize_uninit();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_fast_untilize.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif

#ifdef LLK_PROFILER
    constexpr std::uint32_t NUM_GUARD = 0;
#else
    const std::uint32_t NUM_GUARD = params.NUM_GUARD_TILES;
#endif

    const std::uint32_t total_tiles = FULL_RT_DIM * FULL_CT_DIM;
    const std::uint32_t tile_bytes  = GET_L1_HEADERLESS_TILE_SIZE(formats.pack_dst) << 4;

    if (NUM_GUARD > 0)
    {
        for (std::uint32_t g = 0; g < NUM_GUARD; g++)
        {
            volatile std::uint16_t* guard = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + g]);
            for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
            {
                guard[i] = 0xACAF;
            }
        }
    }

    {
        ZONE_SCOPED("INIT")
        _llk_pack_dest_init_<FAST_UNTILIZE_INTERNAL_DEST_SYNC, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
            formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));
        ckernel::_llk_pack_fast_untilize_init_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }

        if constexpr (FAST_UNTILIZE_SINGLE_UNIT)
        {
            constexpr std::uint32_t unit_dim = FULL_CT_DIM;
            std::uint32_t prev_pack_unit_dim = 0;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    const std::uint32_t tile_row_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, rt * FULL_CT_DIM * TILE_R_DIM * TILE_C_DIM) / 16;
                    const std::uint32_t chunk_address       = L1_ADDRESS(buffer_Res[0]) + tile_row_offset_16B;

                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    ckernel::_llk_pack_fast_untilize_block_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM>(chunk_address, unit_dim, prev_pack_unit_dim);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DEST_SYNC, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        else
        {
            std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
            const std::uint32_t units_per_row = ckernel::fast_untilize_decompose_row(FULL_CT_DIM, unit_dims);
            std::uint32_t prev_pack_unit_dim  = 0;
            const std::uint32_t output_row_stride_16B = SCALE_DATUM_SIZE(formats.pack_dst, FULL_CT_DIM * TILE_C_DIM) / 16;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
                {
                    const std::uint32_t tile_row_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, rt * FULL_CT_DIM * TILE_R_DIM * TILE_C_DIM) / 16;
                    const std::uint32_t tile_row_address    = L1_ADDRESS(buffer_Res[0]) + tile_row_offset_16B;
                    std::uint32_t chunk_col                 = 0;
                    for (std::uint32_t u = 0; u < units_per_row; u++)
                    {
                        const std::uint32_t unit_dim         = unit_dims[u];
                        const std::uint32_t chunk_offset_16B = SCALE_DATUM_SIZE(formats.pack_dst, chunk_col * TILE_C_DIM) / 16;
                        const std::uint32_t chunk_address    = tile_row_address + chunk_offset_16B;

                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                        {
                            _llk_packer_wait_for_math_done_();
                        }
                        ckernel::_llk_pack_fast_untilize_block_strided_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM, FULL_CT_DIM>(
                            chunk_address, unit_dim, prev_pack_unit_dim, output_row_stride_16B);
                        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                        {
                            _llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DEST_SYNC, is_fp32_dest_acc_en>();
                        }
                        chunk_col += unit_dim;
                    }
                }
            }
        }
        PROFILER_SYNC();
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            if (NUM_GUARD == 0)
            {
                return;
            }
        }
    }
    {
        ZONE_SCOPED("UNINIT")
        ckernel::_llk_pack_fast_untilize_uninit_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM, FULL_CT_DIM>(formats.pack_src);
    }

    if (NUM_GUARD > 1)
    {
        volatile std::uint16_t* result_tile = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + NUM_GUARD - 1]);
        for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
        {
            result_tile[i] = 0;
        }
        result_tile[0] = 0x4680;

        for (std::uint32_t g = 0; g < NUM_GUARD - 1; g++)
        {
            volatile std::uint16_t* guard = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + g]);
            std::uint32_t corrupted       = 0;
            for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
            {
                if (guard[i] != 0xACAF)
                {
                    corrupted++;
                }
            }
            result_tile[g + 1] = static_cast<std::uint16_t>(corrupted);
        }
    }
}

#endif
