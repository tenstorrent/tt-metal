// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

#if defined(LLK_PROFILER)
uint32_t loop_factor = 1024;
#else
uint32_t loop_factor = 1;
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

// the kernel algorithm is the same across all threads, with the only difference being the particular calls made to the llks
// the high level algorithm is explained here as well as in the metal compute api where it is copied from
// the kernel tilizes the input tensor of arbitrary shape (divisible by tile dimensions in both axes)
// tensor shape is defined by BLOCK_RT_DIM and BLOCK_CT_DIM which represent the number of rows and columns of tiles respectively
// they are calculated as BLOCK_RT_DIM = TENSOR_HEIGHT / TILE_R_DIM and BLOCK_CT_DIM = TENSOR_WIDTH / TILE_C_DIM
// the first two loops are part of the kernel, providing looping for the performance measurement and calling the high level algorithm for each row of tiles
// everything inside the "for (uint32_t i = 0; i < BLOCK_RT_DIM; i++)" loop implements the actual high level algorithm as used in the metal compute api

// the goal of the high level algorithm is to break down the BLOCK_CT_DIM of the input tensor into a sequence of llk calls
// using available unit_dim primitives while maximizing the utilization of the destination buffer

// principle of operation is as follows:
// BLOCK_CT_DIM == 1: Single call with unit_dim == 1
// BLOCK_CT_DIM == 2: Single call with unit_dim == 2
// BLOCK_CT_DIM == 3: Single call with unit_dim == 3
// BLOCK_CT_DIM > 3 && BLOCK_CT_DIM % 2 == 0: BLOCK_CT_DIM / 2 calls with unit_dim == 2
// BLOCK_CT_DIM > 3 && BLOCK_CT_DIM % 2 == 1: (BLOCK_CT_DIM - 3) / 2 calls with unit_dim == 2, plus one call with unit_dim == 3
// it is good to note that only unit_dim == 2 is called multiple times and/or with num_units > 1
// meaning that unit_dims 1 and 3 will only appear as the last call in the last dest bank

// each iteration of the while loop represents processing of a single dest bank
// the idea is to process as many tiles as possible in each iteration, except the last two
// where the aim is to balance the number of tiles processed between the two dest banks to minimize the perf hit
// depending on the number of remaining tiles, algorithm will decide how many tiles to process
// if remaining_tiles > 2 * dest_size, there will be at least two more dest banks to process so the algorithm is free to process as much tiles as possible
// if remaining_tiles > dest_size, there will be at least one more dest bank to process so the algorithm will aim to split the remaining tiles eavenly
// "even_remainder" is calculated by taking half the tiles and rounding it up to be even as all dest banks except the last only use unit_dim 2
// for all valid inputs ([5, 8] and [9, 16], depending on the dest_size) the number of tiles processed in the last two dest banks will be within 2 of each other
// and number of tiles left in the last dest bank will be at least 2
// in the last iteration there are three distinct cases:
// if the number of remaining tiles is even or unit_dim == 1, the algorithm will process all remaining tiles in a single call
// if the number of remaining tiles is 3, the algorithm will process them in a single call with unit_dim == 3
// if the number of remaining tiles is odd and greater than 3, the algorithm will process all but the last three tiles in a single call with unit_dim == 2
// followed by a call with unit_dim == 3 for the last three tiles

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_fast_tilize_hw_configure_<is_fp32_dest_acc_en>(formats.unpack_src, formats.unpack_dst);
        _llk_unpack_fast_tilize_init_(formats.unpack_dst, BLOCK_CT_DIM);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (uint32_t loop = 0; loop < loop_factor; loop++)
        {
            for (uint32_t i = 0; i < BLOCK_RT_DIM; i++)
            {
                uint32_t read_offset = i * BLOCK_CT_DIM * TILE_R_DIM;

                uint32_t packed_tiles    = 0;
                uint32_t remaining_tiles = BLOCK_CT_DIM;
                uint32_t dest_size       = is_fp32_dest_acc_en ? 4 : 8;
                uint32_t unit_dim        = BLOCK_CT_DIM == 1 ? 1 : 2;
                uint32_t num_units       = dest_size / unit_dim;

                while (packed_tiles < BLOCK_CT_DIM)
                {
                    uint32_t tile_index = read_offset + packed_tiles;
                    if (remaining_tiles > 2 * dest_size)
                    {
                        _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), tile_index, formats.unpack_src, unit_dim, num_units, BLOCK_CT_DIM);
                        packed_tiles += dest_size;
                        remaining_tiles -= dest_size;
                    }
                    else if (remaining_tiles > dest_size)
                    {
                        uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
                        num_units               = even_remainder / unit_dim;
                        _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), tile_index, formats.unpack_src, unit_dim, num_units, BLOCK_CT_DIM);
                        packed_tiles += even_remainder;
                        remaining_tiles -= even_remainder;
                    }
                    else
                    {
                        if (remaining_tiles % 2 == 0 || unit_dim == 1)
                        {
                            num_units = remaining_tiles / unit_dim;
                            _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), tile_index, formats.unpack_src, unit_dim, num_units, BLOCK_CT_DIM);
                        }
                        else if (remaining_tiles == 3)
                        {
                            _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), tile_index, formats.unpack_src, 3, 1, BLOCK_CT_DIM);
                        }
                        else
                        {
                            num_units = (remaining_tiles - 3) / unit_dim;
                            _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), tile_index, formats.unpack_src, unit_dim, num_units, BLOCK_CT_DIM);
                            _llk_unpack_fast_tilize_block_(L1_ADDRESS(buffer_A[0]), tile_index + remaining_tiles - 3, formats.unpack_src, 3, 1, BLOCK_CT_DIM);
                        }
                        packed_tiles += remaining_tiles;
                        remaining_tiles = 0;
                    }
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_(formats.math, formats.math);
        _llk_math_fast_tilize_init_(formats.math, BLOCK_CT_DIM == 1 ? 1 : 2);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (uint32_t loop = 0; loop < loop_factor; loop++)
        {
            for (uint32_t i = 0; i < BLOCK_RT_DIM; i++)
            {
                uint32_t packed_tiles    = 0;
                uint32_t remaining_tiles = BLOCK_CT_DIM;
                uint32_t dest_size       = is_fp32_dest_acc_en ? 4 : 8;
                uint32_t unit_dim        = BLOCK_CT_DIM == 1 ? 1 : 2;
                uint32_t num_units       = dest_size / unit_dim;

                while (packed_tiles < BLOCK_CT_DIM)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

                    if (remaining_tiles > 2 * dest_size)
                    {
                        _llk_math_fast_tilize_block_(0, formats.math, unit_dim, num_units);
                        packed_tiles += dest_size;
                        remaining_tiles -= dest_size;
                    }
                    else if (remaining_tiles > dest_size)
                    {
                        uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
                        num_units               = even_remainder / unit_dim;
                        _llk_math_fast_tilize_block_(0, formats.math, unit_dim, num_units);
                        packed_tiles += even_remainder;
                        remaining_tiles -= even_remainder;
                    }
                    else
                    {
                        if (remaining_tiles % 2 == 0 || unit_dim == 1)
                        {
                            num_units = remaining_tiles / unit_dim;
                            _llk_math_fast_tilize_block_(0, formats.math, unit_dim, num_units);
                        }
                        else if (remaining_tiles == 3)
                        {
                            _llk_math_fast_tilize_block_(0, formats.math, 3, 1);
                        }
                        else
                        {
                            num_units = (remaining_tiles - 3) / unit_dim;
                            _llk_math_fast_tilize_block_(0, formats.math, unit_dim, num_units);
                            _llk_math_fast_tilize_block_(remaining_tiles - 3, formats.math, 3, 1);
                        }
                        packed_tiles += remaining_tiles;
                        remaining_tiles = 0;
                    }

                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel()
{
    uint32_t use_32bit_dest = formats.unpack_dst == static_cast<std::underlying_type_t<DataFormat>>(DataFormat::Tf32);
    {
        ZONE_SCOPED("INIT")
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, false>();
        _llk_pack_fast_tilize_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst);
        _llk_pack_fast_tilize_init_<DstSync::SyncHalf>(use_32bit_dest, formats.pack_dst, BLOCK_CT_DIM == 1 ? 1 : 2);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (uint32_t loop = 0; loop < loop_factor; loop++)
        {
            for (uint32_t i = 0; i < BLOCK_RT_DIM; i++)
            {
                uint32_t write_offset = i * BLOCK_CT_DIM;

                uint32_t packed_tiles    = 0;
                uint32_t remaining_tiles = BLOCK_CT_DIM;
                uint32_t dest_size       = is_fp32_dest_acc_en ? 4 : 8;
                uint32_t unit_dim        = BLOCK_CT_DIM == 1 ? 1 : 2;
                uint32_t num_units       = dest_size / unit_dim;

                while (packed_tiles < BLOCK_CT_DIM)
                {
                    _llk_packer_wait_for_math_done_();

                    uint32_t tile_index = write_offset + packed_tiles;
                    if (remaining_tiles > 2 * dest_size)
                    {
                        _llk_pack_fast_tilize_block_(0, L1_ADDRESS(buffer_Res[tile_index]), unit_dim, num_units);
                        packed_tiles += dest_size;
                        remaining_tiles -= dest_size;
                    }
                    else if (remaining_tiles > dest_size)
                    {
                        uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
                        num_units               = even_remainder / unit_dim;
                        _llk_pack_fast_tilize_block_(0, L1_ADDRESS(buffer_Res[tile_index]), unit_dim, num_units);
                        packed_tiles += even_remainder;
                        remaining_tiles -= even_remainder;
                    }
                    else
                    {
                        if (remaining_tiles % 2 == 0 || unit_dim == 1)
                        {
                            num_units = remaining_tiles / unit_dim;
                            _llk_pack_fast_tilize_block_(0, L1_ADDRESS(buffer_Res[tile_index]), unit_dim, num_units);
                        }
                        else if (remaining_tiles == 3)
                        {
                            _llk_pack_fast_tilize_block_(0, L1_ADDRESS(buffer_Res[tile_index]), 3, 1);
                        }
                        else
                        {
                            num_units = (remaining_tiles - 3) / unit_dim;
                            _llk_pack_fast_tilize_block_(0, L1_ADDRESS(buffer_Res[tile_index]), unit_dim, num_units);
                            _llk_pack_fast_tilize_block_(remaining_tiles - 3, L1_ADDRESS(buffer_Res[tile_index + remaining_tiles - 3]), 3, 1);
                        }
                        packed_tiles += remaining_tiles;
                        remaining_tiles = 0;
                    }

                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst);
}

#endif
