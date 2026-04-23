// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// BH full fast-tilize: unpack + math + pack → standard tilized output.
// Supports arbitrary ct_dim >= 2 via stride-preserving unit_dim={4,2,3}.
// Width 1 uses standard tilize fallback.
// Supports PERF_RUN_TYPE for performance measurements.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

#ifndef PERF_RUN_TYPE
#define PERF_RUN_TYPE PerfRunType::L1_TO_L1
#endif
#include "perf.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Row decomposition: split ct_dim into {4, 2, 3} unit sequence.
// Each unit produces the same DEST layout (stale gaps for unused W positions).
constexpr std::uint32_t MAX_UNITS_PER_ROW = 16;

inline std::uint32_t decompose_row(const std::uint32_t ct_dim, std::uint32_t unit_dims[MAX_UNITS_PER_ROW])
{
    std::uint32_t n4  = ct_dim / 4;
    std::uint32_t rem = ct_dim % 4;
    std::uint32_t idx = 0;

    if (rem == 1 && n4 > 0)
    {
        n4--;
        rem = 5; // rewrite 4+1 as 2+3
    }

    for (std::uint32_t i = 0; i < n4; i++)
    {
        unit_dims[idx++] = 4;
    }

    if (rem == 2)
    {
        unit_dims[idx++] = 2;
    }
    else if (rem == 3)
    {
        unit_dims[idx++] = 3;
    }
    else if (rem == 5)
    {
        unit_dims[idx++] = 2;
        unit_dims[idx++] = 3;
    }
    return idx;
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const Operand& buffer_A          = params.buffer_A;
#endif

    std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
    std::uint32_t units_per_row = decompose_row(BLOCK_CT_DIM, unit_dims);

    // Width 1 fallback
    if (BLOCK_CT_DIM == 1)
    {
        {
            ZONE_SCOPED("INIT")
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
            _llk_unpack_tilize_init_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false);
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                    {
                        _llk_unpack_tilize_(
                            L1_ADDRESS(buffer_A[0]), row * BLOCK_CT_DIM * TILE_R_DIM, formats.unpack_A_src, formats.unpack_A_dst, 0, FACE_R_DIM, 4, false);
                    }
                }
            }
        }
        {
            ZONE_SCOPED("UNINIT")
            _llk_unpack_tilize_uninit_(formats.unpack_A_dst, 4, FACE_R_DIM);
        }
        return;
    }

    {
        ZONE_SCOPED("INIT")
        // Use the inferred unpack_A_dst directly so test_fast_tilize_full covers the
        // Float32-input path end-to-end (including the Tf32-SrcA case that production
        // conv2d hits). Earlier this was forced to Float16_b as a workaround; now the
        // LLK is expected to handle fp32 input correctly.
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, BLOCK_CT_DIM, unit_dims[0]);
        // Base address is programmed per-call inside _llk_unpack_fast_tilize_block_
        // via _llk_unpack_configure_single_address_ (respects current cfg context).
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Each unit produces 4 dvalids regardless of unit_dim
            std::uint32_t total_units = BLOCK_RT_DIM * units_per_row;
            return _perf_unpack_loop_set_valid<true, false>(total_units * 4 * LOOP_FACTOR);
        }
        else
        {
            // Per-row per-chunk block calls (mirrors tt-metal API compute/tilize.h
            // fast_tilize_block which iterates chunks of a single row). Each call
            // processes one chunk at col_offset of one row; row base advances via L1.
            std::uint32_t prev_chunk = unit_dims[0];
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                {
                    std::uint32_t col_offset = 0;
                    for (std::uint32_t u = 0; u < units_per_row; u++)
                    {
                        std::uint32_t chunk = unit_dims[u];
                        if (chunk != prev_chunk)
                        {
                            _llk_unpack_fast_tilize_reinit_xdim_(chunk);
                            prev_chunk = chunk;
                        }
                        _llk_unpack_fast_tilize_block_(
                            L1_ADDRESS(buffer_A[row * BLOCK_CT_DIM]),
                            0 /*tile_index unused*/,
                            formats.unpack_A_src,
                            chunk /*unit_dim*/,
                            1 /*num_units: one chunk*/,
                            chunk /*ct_dim: units_per_row=1, num_rows=1*/,
                            4,
                            col_offset);
                        col_offset += chunk;
                    }
                }
            }
        }
    }
    {
        ZONE_SCOPED("UNINIT")
        _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
#endif
    constexpr std::uint32_t unit_dim = 4;

    std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
    std::uint32_t units_per_row = decompose_row(BLOCK_CT_DIM, unit_dims);

    // Width 1 fallback
    if (BLOCK_CT_DIM == 1)
    {
        {
            ZONE_SCOPED("INIT")
            _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
            _llk_math_eltwise_unary_datacopy_init_<A2D, is_fp32_dest_acc_en, BroadcastType::NONE, true>(4, formats.math);
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    _llk_math_eltwise_unary_datacopy_<A2D, DstSync::SyncHalf, is_fp32_dest_acc_en>(0, formats.math, formats.math, 4);
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        {
            ZONE_SCOPED("UNINIT")
        }
        return;
    }

    std::uint32_t num_units_total = BLOCK_RT_DIM * units_per_row;

    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(formats.math, unit_dim);
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // Release one section_done per unit so pack can run all units
            for (std::uint32_t i = 0; i < num_units_total; i++)
            {
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return _perf_math_loop_clear_valid<true, false>(num_units_total * 4 * LOOP_FACTOR);
        }
        else
        {
            // Math is completely unchanged — same MOP for all unit_dims.
            // Each unit consumes 4 dvalids and fills a full DEST half-bank.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t u = 0; u < num_units_total; u++)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(0, formats.math, unit_dim, 1, 4);
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
    }
    {
        ZONE_SCOPED("UNINIT")
        _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_pack_fast_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const Operand& buffer_Res        = params.buffer_Res;
#endif

    std::uint32_t unit_dims[MAX_UNITS_PER_ROW];
    std::uint32_t units_per_row = decompose_row(BLOCK_CT_DIM, unit_dims);

    const std::uint32_t total_tiles = BLOCK_CT_DIM * BLOCK_RT_DIM;
    const std::uint32_t tile_bytes  = GET_L1_HEADERLESS_TILE_SIZE(formats.pack_dst) << 4;

    // Guard tile sentinel: gated by NUM_GUARD_TILES runtime param.
    // Perf tests compile with LLK_PROFILER and don't have this param in their struct.
#ifdef LLK_PROFILER
    constexpr std::uint32_t NUM_GUARD = 0;
#else
    const std::uint32_t NUM_GUARD = params.NUM_GUARD_TILES;
#endif
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

    // Width 1 fallback
    if (BLOCK_CT_DIM == 1)
    {
        {
            ZONE_SCOPED("INIT")
            _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));
            _llk_pack_init_<false, false, true>(formats.pack_src, formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4, false, false, 1, false);
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, L1_ADDRESS(buffer_Res[row * BLOCK_CT_DIM]));
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        {
            ZONE_SCOPED("UNINIT")
        }
        // Fall through to sentinel check below
    }
    else
    {
        std::uint32_t num_units_total = BLOCK_RT_DIM * units_per_row;

        {
            ZONE_SCOPED("INIT")
            _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));
            _llk_pack_fast_tilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, formats.pack_dst, unit_dims[0], 4, formats.pack_src);
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
            {
                return;
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t i = 0; i < num_units_total * LOOP_FACTOR; i++)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
                return;
            }
            else
            {
                std::uint32_t prev_udim = unit_dims[0];

                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                    {
                        std::uint32_t col_offset = 0;

                        for (std::uint32_t u = 0; u < units_per_row; u++)
                        {
                            std::uint32_t udim = unit_dims[u];

                            if (udim != prev_udim)
                            {
                                _llk_pack_fast_tilize_reinit_unit_dim_(formats.pack_dst, udim);
                                prev_udim = udim;
                            }

                            _llk_packer_wait_for_math_done_();
                            std::uint32_t tile_offset = row * BLOCK_CT_DIM + col_offset;
                            _llk_pack_fast_tilize_block_(0, L1_ADDRESS(buffer_Res[tile_offset]), udim, 1, 4);
                            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

                            col_offset += udim;
                        }
                    }
                }
            }
        }
        {
            ZONE_SCOPED("UNINIT")
            _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, FACE_R_DIM, 4, formats.pack_src);
        }

    } // end else (fast tilize path)

    // Post-tilize: check sentinel integrity on raw uint16 level.
    if (NUM_GUARD > 1)
    {
        volatile std::uint16_t* result_tile = reinterpret_cast<volatile std::uint16_t*>(buffer_Res[total_tiles + NUM_GUARD - 1]);
        for (std::uint32_t i = 0; i < tile_bytes / 2; i++)
        {
            result_tile[i] = 0;
        }
        result_tile[0] = 0x4680; // marker

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
