// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hand-coded parameter encoding for various GPR mappings
namespace ckernel
{

// Common GPR mapping across all threads
struct p_gpr
{
    constexpr static uint ZERO     = 0;  // Always stores 0
    constexpr static uint DBG_RESERVED = 1;  // Reserved for future use
    constexpr static uint DBG_MSG =  2;  // Firmware debug message
    constexpr static uint DBG_CKID = 3;  // Ckernel ID
};

// Unpack GPR thread
struct p_gpr_unpack
{
    constexpr static uint OPERAND_BASE_ADDR = 4;   // Operand base address used by zero buffer function
    constexpr static uint OPERAND_OFFSET_ADDR = 5;   // Operand offset address used by zero buffer function
    constexpr static uint ZERO_0            = 8;   // Zero data
    constexpr static uint ZERO_1            = 9;   // Zero data
    constexpr static uint ZERO_2            = 10;   // Zero data
    constexpr static uint ZERO_3            = 11;   // Zero data
    constexpr static uint TMP0              = 12;   // Temp data
    constexpr static uint TMP1              = 13;   // Temp data
    constexpr static uint TILE_SIZE         = 14;   // Tile size stored for untilize llk
    constexpr static uint TILE_OFFSET       = 15;   // Tile offset address used by untilize llk
    constexpr static uint PERF_FIRST_UNP_LO = 32;   // timestamp for first-unpack-instruction (low 32b)
    constexpr static uint PERF_FIRST_UNP_HI = 33;   // timestamp for first-unpack-instruction (high 32b)
    constexpr static uint PERF_UNPACK_NUM_TILES_0 = 34;   // num tiles for input operands 0-1
    constexpr static uint PERF_UNPACK_NUM_TILES_1 = 35;   // num tiles for input operands 2-3
    constexpr static uint PERF_UNPACK_NUM_TILES_2 = 36;   // num tiles for input operands 4-5
    constexpr static uint PERF_UNPACK_NUM_TILES_3 = 37;   // num tiles for input operands 6-7
    constexpr static uint FACE_DIM_16x16    = 40;   // Holds face dimension (16x16)
    constexpr static uint FACE_DIM_8x16     = 41;   // Holds face dimension (8x16)
    constexpr static uint FACE_DIM_4x16     = 42;   // Holds face dimension (4x16)
    constexpr static uint FACE_DIM_2x16     = 43;   // Holds face dimension (2x16)
    constexpr static uint FACE_DIM_1x16     = 44;   // Holds face dimension (1x16)
    constexpr static uint SR_UNPACK_TILIZER_STATE_0 = 54;  // Save unpack state before tilizer is enabled for quick restore
    constexpr static uint SR_UNPACK_TILIZER_STATE_1 = 55;
    constexpr static uint SR_UNPACK_UNTILIZER_STATE_0 = 56;  // Save unpack state before tilizer is enabled for quick restore
    constexpr static uint SR_UNPACK_UNTILIZER_STATE_1 = 57;
    constexpr static uint SR_UNPACK_UNTILIZER_STATE_2 = 58;
    constexpr static uint SR_UNPACK_UNTILIZER_STATE_3 = 59;
};

// Math GPR thread
struct p_gpr_math
{
    constexpr static uint PERF_DBUS_CNTL    = 4;   // Control debug bus perf counter selection
    constexpr static uint PERF_MEM_DUMP_CNTL_CLEAR= 5;   // Clear write to memory flag
    constexpr static uint PERF_MEM_DUMP_CNTL_SET= 6;   // Set write to memory flag
    constexpr static uint PERF_CNT_START    = 7;   // Start perf counter
    constexpr static uint PERF_CNT_STOP     = 8;   // Stop perf counter
    constexpr static uint PERF_EPOCH_BASE_ADDR= 9; // Perf event ID
    constexpr static uint PERF_EPOCH_OFFSET = 10; // The offset address for event id
    constexpr static uint PERF_BUF_BASE     = 11; // Base address for perf buffer
    constexpr static uint PERF_EVENT_ID     = 12; // Event id for math thread events
    constexpr static uint PERF_EVENT_OFFSET = 13; // The offset address for event id
    constexpr static uint DEST_OP0_BASE     = 48;  // dest base for sfpu op0
    constexpr static uint DEST_OP1_BASE     = 49;  // dest base for sfpu op1
    constexpr static uint DEST_REGW_OFFSET  = 50;  // dest rwc base (1st set)
    constexpr static uint DEST_REGW_INCR    = 51;  // dest rwc incr (1st set)
    constexpr static uint DEST_REGW_OFFSET2 = 52;  // dest rwc base (2nd set)
    constexpr static uint DEST_REGW_INCR2   = 53;  // dest rwc incr (2nd set)
    constexpr static uint TMP0              = 60;
    constexpr static uint NUM_DRAM_REQS     = 61;
};

// Pack GPR thread
struct p_gpr_pack
{
    constexpr static uint DEST_OFFSET_LO = 4;   // dest lower bank offsets
    constexpr static uint DEST_OFFSET_HI = 8;   // dest upper bank offsets
    constexpr static uint OUTPUT_ADDR    = 12;  // output address that packer is writing to
    constexpr static uint TILE_HEADER    = 16;  // tile header - ID + tile size

    constexpr static uint TMP_DEST_OFFSET   = 20; // Temp var which holds tile offset in dest
    constexpr static uint NUM_MSGS_RECEIVED = 24;  // holds tile count and word size
    constexpr static uint ONE_MSG_RECEIVED  = 25;  // by default holds 1 tile count and word size for streaming per tile
    constexpr static uint HEADER_ADDR       = 26;  // Holds the address of the header (used by pack shift kernel only)
    constexpr static uint TMP0              = 28;   // Temp data
    constexpr static uint TMP1              = 29;   // Temp data
    constexpr static uint PACK_STREAM_SYNC  = 32;  // sync between pack and output stream [32:63]
};

} // namespace ckernel
