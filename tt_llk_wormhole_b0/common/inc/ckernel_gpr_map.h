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
    constexpr static uint TILE_SIZE         = 14;   // Tile size
    constexpr static uint TILE_OFFSET       = 15;   // Tile offset
    constexpr static uint L1_BUFFER_ADDR    = 17;   // Holds address of fixed l1 buffer used for reduce in1 
    constexpr static uint TMP_LO            = 18;   // Temp data. Upper 16-bits always 0
    constexpr static uint TMP_HI            = 19;   // Temp data. Lower 16-bits always 0
    constexpr static uint PERF_FIRST_UNP_LO = 32;   // timestamp for first-unpack-instruction (low 32b)
    constexpr static uint PERF_FIRST_UNP_HI = 33;   // timestamp for first-unpack-instruction (high 32b)
    constexpr static uint TILE_SIZE_A       = 36;   // Holds tile size for unpacker 0
    constexpr static uint TILE_SIZE_B       = 37;   // Holds tile size for unpacker 1
    constexpr static uint KT_DIM            = 38;   // Holds matmul kt_dim
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
    constexpr static uint PERF_EPOCH_OFFSET = 10; // The offset address for epoch variables
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

    constexpr static uint TEMP_TILE_OFFSET  = 20; // Temp var which holds tile offset in dest
    constexpr static uint NUM_MSGS_RECEIVED = 24;  // holds tile count and word size
    constexpr static uint ONE_MSG_RECEIVED  = 25;  // by default holds 1 tile count and word size for streaming per tile
    constexpr static uint HEADER_ADDR       = 26;  // Holds the address of the header (used by pack shift kernel only)
    constexpr static uint TMP0              = 28;   // Temp data
    constexpr static uint TMP1              = 29;   // Temp data
    constexpr static uint TMP_LO            = 30;   // Temp data, upper 16-bit always 0
    constexpr static uint TMP_HI            = 31;   // Temp data, lower 16-bit always 0
    constexpr static uint PACK_STREAM_SYNC  = 32;  // sync between pack and output stream [32:63]
    constexpr static uint EXP0_SEC_SIZE_BFP   = 52;  // pack0,1,2,3 exp section size for bfp8,4,2
    constexpr static uint EXP1_SEC_SIZE_BFP8  = 53;  // pack1 exp section size for bfp8
    constexpr static uint EXP2_SEC_SIZE_BFP8  = 54;  // pack2 exp section size for bfp8
    constexpr static uint EXP3_SEC_SIZE_BFP8  = 55;  // pack2 exp section size for bfp8
    constexpr static uint EXP1_SEC_SIZE_BFP4  = 57;  // pack1 exp section size for bfp4
    constexpr static uint EXP2_SEC_SIZE_BFP4  = 58;  // pack2 exp section size for bfp4
    constexpr static uint EXP3_SEC_SIZE_BFP4  = 59;  // pack3 exp section size for bfp4
    constexpr static uint EXP1_SEC_SIZE_BFP2  = 61;  // pack1 exp section size for bfp2
    constexpr static uint EXP2_SEC_SIZE_BFP2  = 62;  // pack2 exp section size for bfp2
    constexpr static uint EXP3_SEC_SIZE_BFP2  = 63;  // pack3 exp section size for bfp2
};

} // namespace ckernel
