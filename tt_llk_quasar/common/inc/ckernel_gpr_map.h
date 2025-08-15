// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Hand-coded parameter encoding for various GPR mappings
namespace ckernel
{

// Common GPR mapping across all threads
struct p_gpr
{
    constexpr static uint ZERO     = 0; // Always stores 0
    constexpr static uint RBASE    = 1; // Ring pointer base
    constexpr static uint ROFFSET  = 2; // Ring pointer offset
    constexpr static uint DBG_CKID = 3; // Ckernel ID
};

// Unpack GPR thread
struct p_gpr_unpack
{
    constexpr static uint X_DIM_TOP     = 50; // top halo params
    constexpr static uint X_DIM_BOT     = 51; // bottom halo params
    constexpr static uint WEIGHT_OFFSET = 50; //
    constexpr static uint WEIGHT_INCR   = 51; //
    constexpr static uint TEMPORARY     = 52; //
    constexpr static uint DATUM_CNT_TOP = 56; // top halo params
    constexpr static uint DATUM_CNT_BOT = 57; // bottom halo params
};

// Math GPR thread
struct p_gpr_math
{
    constexpr static uint TEMP0             = 4;  // dest rwc base (1st set)
    constexpr static uint DEST_REGW_OFFSET  = 50; // dest rwc base (1st set)
    constexpr static uint DEST_REGW_INCR    = 51; // dest rwc incr (1st set)
    constexpr static uint DEST_REGW_OFFSET2 = 52; // dest rwc base (2nd set)
    constexpr static uint DEST_REGW_INCR2   = 53; // dest rwc incr (2nd set)
};

// Pack GPR thread
struct p_gpr_pack
{
    constexpr static uint TEMP0     = 4;
    constexpr static uint TEMP1     = 5;
    constexpr static uint TEMP2     = 6;
    constexpr static uint SYNC_FLAG = 7;

    // Standard 1p/2p/4p/strips packers
    constexpr static uint OUTPUT_ADDR    = 12; // output address that packer is writing to
    constexpr static uint MSGINFO_ADDR   = 16; // msg info address read from overlay
    constexpr static uint FORCE_MAX_XY   = 39; // 32b aligned read for force_max_xy reg
    constexpr static uint TILE_HEADER    = 40; // tile header - ID + tile size
    constexpr static uint TILE_METADATA  = 41; // tile meta data + format and compression flag
    constexpr static uint TILE_ZERO_MASK = 42; // zero mask for the output tile
    constexpr static uint TILE_RESERVED  = 43; // reserved
    constexpr static uint DEST_OFFSET_HI = 56; // dest upper bank offsets
    constexpr static uint DEST_OFFSET_LO = 60; // dest lower bank offsets
    constexpr static uint FLUSH_COUNTERS = 56; // flush pack counters
    constexpr static uint FLUSH_OFFSET   = 60; // flush dest read offsets

    // Special packer spilling to L1 scratch buffer
    constexpr static uint OUTPUT_FORMAT        = 50; // output buffer format
    constexpr static uint SCRATCH_FORMAT       = 51; // scratch buffer format
    constexpr static uint TILE_STITCHER_HEADER = 52; // used to store tile header offset
    constexpr static uint TILE_STITCHER_GAP    = 53; // used to store tile stitching gap
    constexpr static uint TEMPORARY            = 54; //
    constexpr static uint PACK_COUNTERS        = 55; // only used if counter reprogramming is required
};

struct p_params
{
    // Param array mappings
    constexpr static uint CMD_BASE = 0;

    // Param load settings
    constexpr static uint DST_COPY   = 0; // L1 copy into localmem then access (Requires repeated use to amortize L1 copy cost)
    constexpr static uint SRC_DIRECT = 1; // L1 direct access of params (Fastest for single use or user managed local copies)
};

} // namespace ckernel
