// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSIX_TYPES_H_INCLUDED
#define TENSIX_TYPES_H_INCLUDED

#include <cstdint>

#ifndef TENSIX_FIRMWARE
#include <boost/format.hpp>
#endif

//
//  tensix_types.h
//  This file contains tensix structures used by RISCV firmware and test-bench/tests
//
//  Copyright © 2018 Tenstorrent. All rights reserved.
//

/////////////
// Global enums and defines
////////////
typedef enum
{
    XMOV_L0_TO_L1 = 0,
    XMOV_L1_TO_L0 = 1,
    XMOV_L0_TO_L0 = 2,
    XMOV_L1_TO_L1 = 3,
} xmov_direction_t;

typedef enum
{
    TDMA_MOVER0 = 0,
    TDMA_MOVER1 = 1
} tdma_mover_id_t;

typedef enum
{
    MATH_HF   = 1,
    MATH_AUTO = 2,
    MATH_LF   = 4
} math_fidelity_t;

typedef enum
{
    RELU_NONE   = 0,
    RELU_PLAIN  = 1,
    RELU_THRESH = 2,
    RELU_MAX    = 3
} relu_mode_t;

typedef enum
{
    STOCH_RND_NONE   = 0,
    STOCH_RND_FPU    = 1,
    STOCH_RND_GASKET = 2,
    STOCH_RND_PACKER = 4
} stochastic_round_settings_t;

/////////////
// TDMA Registers
////////////
typedef struct
{
    uint32_t row_section_size : 16;
    uint32_t exp_section_size : 16;
    uint32_t tile_dst_addr    : 32;
    uint32_t uncompressed     : 1;
    uint32_t reserved_0       : 3;
    uint32_t out_data_format  : 2;
    uint32_t reserved_1       : 2;
    uint32_t in_data_format   : 2;
    uint32_t reserved_2       : 22;
    uint32_t reserved_3       : 32;
} packer_config_t; // 16B

typedef struct
{
    uint32_t rd_ptr;
    uint32_t wr_ptr;
    uint32_t rsvd0;
    uint32_t rsvd1;
#ifndef TENSIX_FIRMWARE
    operator std::string() const
    {
        return (boost::format("Fifo Control: rd_ptr(0x%08x) wr_ptr(0x%08x)") % rd_ptr % wr_ptr).str();
    }
#endif
} fifo_ctl_t;

typedef struct
{
    uint32_t val[4];
    packer_config_t f;
} packer_config_u;

typedef struct
{
    uint32_t src_addr   : 32;
    uint32_t dst_addr   : 32;
    uint32_t xfer_size  : 32;
    uint32_t xfer_dir   : 2;
    uint32_t reserved_0 : 30;
} mover_config_t; // 16B

typedef struct
{
    uint32_t val[4];
    mover_config_t f;
} mover_config_u;

/////////////
// Data section structures
/////////////

/* Paste lines between <LUA> and </LUA> into a lua interpreter

-- <LUA>
txt = [[
*/
// Tile descriptor
typedef struct
{
    unsigned x_dim         : 16;
    unsigned y_dim         : 8;
    unsigned z_dim         : 8;
    unsigned w_dim         : 8;
    unsigned in_fmt        : 5; // We should REALLY consider making this 8 bits
    unsigned out_fmt       : 5; // We should REALLY consider making this 8 bits
    unsigned header_absent : 1;
    unsigned               : 13; // Padding
    unsigned l1_base_addr  : 32;
    unsigned l1_addr_off   : 16;
    unsigned buftab_inc    : 4;
    unsigned               : 12; // Padding
} tile_descriptor_t;             // Unpack configuration

/*
]]
sum = 0
for n in txt:gmatch":%s*(%d+);" do
    sum = sum + tonumber(n)
end
if sum ~= 128 then print "You goofed up your bitfield" end

-- </LUA>

*/

static_assert(sizeof(tile_descriptor_t) == 16, "tile_desc must be 128b!"); // Descriptor must be 128 bits

typedef union
{
    uint32_t val[4];
    tile_descriptor_t f;
} tile_descriptor_u;

struct TileHeader
{
    // occupied part of the 16B line
    std::uint16_t tile_size_16B = 0;
    std::uint16_t reserved_0_mbz : 1;
    std::uint16_t tile_id        : 15;

    std::uint8_t metadata_size_16B = 0;
    std::uint8_t reserved_1        = 0;
    std::uint16_t format           = 0x10; // [3:0] format, 4-uncompress flag.

    std::uint32_t zero_mask  = 0;
    std::uint32_t reserved_3 = 0;

    TileHeader() : reserved_0_mbz(0), tile_id(0)
    {
    }

    bool IsCompressed() const
    {
        return ((format & 0x10) == 0);
    }

#ifndef TENSIX_FIRMWARE
    operator std::string() const
    {
        return (boost::format("TileHeader:tile_id(0x%04x) size16B(0x%04x) zero_mask(0x%08x)") % tile_id % tile_size_16B % zero_mask).str();
    }

    std::size_t size() const
    {
        return 16;
    }

    const void* data() const
    {
        return this;
    }

    typedef std::uint8_t value_type;

    bool operator!=(const TileHeader& rhs) const
    {
        bool result = tile_size_16B != rhs.tile_size_16B || tile_id != rhs.tile_id || metadata_size_16B != rhs.metadata_size_16B || zero_mask != rhs.zero_mask;
        return result;
    }

#endif
};

union TileHeader_u
{
    uint32_t val[4];
    TileHeader header;

    TileHeader_u()
    {
    }
};

static_assert(sizeof(TileHeader) == 16, "TileHeader must be 16B");

struct SectionHeader
{
    // occupied part of the 16B line
    std::uint16_t section_id;
    std::uint16_t section_size;
    std::uint16_t tile_count;

    // unoccupied part of the 16B line
    std::uint16_t reserved[5];

#ifndef TENSIX_FIRMWARE
    operator std::string() const
    {
        return (boost::format("SectionHeader: id(0x%04x) size(0x%04x) tile_count(0x%04x)") % section_id % section_size % tile_count).str();
    }
#endif
};

// Actually it only has to be a multiple of 16B
static_assert(sizeof(SectionHeader) == 16, "struct section_header must be 16 bytes");

static constexpr std::uint32_t TEST_MSG_EN_TENSIX_PM         = 0;
static constexpr std::uint32_t TEST_MSG_DBG_DISABLE          = 1;
static constexpr std::uint32_t TEST_MSG_SET_MAX_EXP_THRESH   = 2;
static constexpr std::uint32_t TEST_MSG_RISC_BP_DISABLE      = 3;
static constexpr std::uint32_t TEST_MSG_SET_RELU_PARAMS      = 4;
static constexpr std::uint32_t TEST_MSG_SET_PRNG_SEED        = 5;
static constexpr std::uint32_t TEST_MSG_RISC_PREFETCHER_CTRL = 6;
static constexpr std::uint32_t TEST_MSG_SYNTH_CKERNEL        = 10;
static constexpr std::uint32_t TEST_MSG_ONE_KERNEL           = 11;
static constexpr std::uint32_t TEST_MSG_WAIT_KERNELS         = 12;
static constexpr std::uint32_t TEST_MSG_WAIT_ONE_KERNEL      = 13;
static constexpr std::uint32_t TEST_MSG_SYNTH_TAG_SRCH       = 14;

static constexpr std::uint32_t COMMAND_QUEUE_SIZE_BYTES_LOG2 = 16;
static constexpr std::uint32_t COMMAND_QUEUE_SIZE_BYTES      = 1 << COMMAND_QUEUE_SIZE_BYTES_LOG2;
static constexpr std::uint32_t COMMAND_SIZE_BYTES_LOG2       = 5;
static constexpr std::uint32_t COMMAND_SIZE_BYTES            = 1 << COMMAND_SIZE_BYTES_LOG2;

static constexpr std::uint32_t DEST_FACE_WIDTH         = 16;
static constexpr std::uint32_t DEST_FACE_HEIGHT        = 16;
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE = 64 * DEST_FACE_HEIGHT;
static constexpr std::uint32_t DEST_REGISTER_HALF_SIZE = DEST_REGISTER_FULL_SIZE / 2;

// static constexpr std::uint32_t SIM_L1_SIZE = 0x16E000;   // 1.5MB - 72KB
static constexpr std::uint32_t SIM_L1_SIZE = 0x180000; // bh is now 180000 per marco
#ifdef TENSIX_FIRMWARE
// static constexpr std::uint32_t L1_SIZE = 0x16E000;   // 1.5MB - 72KB
static constexpr std::uint32_t L1_SIZE = 0x180000; // bh is now 180000 per marco
#else
// static constexpr std::uint32_t L1_SIZE = 0x16E000; // 1.5MB - 72KB
static constexpr std::uint32_t L1_SIZE = 0x180000; // bh is now 180000 per marco
#endif

// Voluntary FIFO alignment so that we can pack fifo address down to 16 bits in the command.
// At 8, we can cover 16MB. The upper limit is 15 because the command queue is at 32K (by default)
// Even though the command queue address never goes into a command, assertions complain if it is misaligned.
// (Hardware only requires 16B alignment.)
static constexpr std::uint32_t FIFO_BASE_ADDRESS_ALIGN_BITS = 9;
static constexpr std::uint32_t FIFO_BASE_ADDRESS_ALIGN      = 1 << FIFO_BASE_ADDRESS_ALIGN_BITS;

enum class DataFormat : std::uint8_t
{
    Float32   = 0,  // E8M23
    Tf32      = 4,  // E8M10                 Stored in L1 in 32b container as : {1b sign, 8b exp, 10b man, 13'b0}
    Float16   = 1,  // E5M10
    Float16_b = 5,  // E8M7
    Fp8R      = 10, // E5M2
    Fp8P      = 16, // E4M3
    MxFp8R    = 18, // E5M2 with block exp
    MxFp8P    = 20, // E4M3 with block exp
    MxFp6R    = 19, // E3M2 with block exp;  Element stored in L1 in 8b container as : {1b sign, 3b exp, 2b man, 2'b0}
    MxFp6P    = 21, // E2M3 with block exp;  Element stored in L1 in 8b container as : {1b sign, 2b exp, 3b man, 2'b0}
    MxFp4     = 22, // E2M1 with block exp
    MxInt8    = 2,  // E0M7 with block exp
    MxInt4    = 3,  // E0M3 with block exp
    MxInt2    = 11, // E0M1 with block exp
    Int32     = 8,
    Int8      = 14,
    Int16     = 9,
    Uint8     = 17,  // Unsigned INT with 8-bit magnitude
    Uint16    = 130, // Unsigned INT with 16-bit magnitude
    // Special-case encodings used only for MXFP4 2x-packed Src Reg Storage :
    MxFp4_2x_A = 27, // store MXFP4 in Src Regs as 2x-packed format with 5-bit exp
    MxFp4_2x_B = 24, // store MXFP4 in Src Regs as 2x-packed format with 8-bit exp
    Int4       = 23,
    Uint4      = 25,
    // Special-case encodings used only for int 2x-packed Src Reg Storage :
    Int8_2x  = 26,
    Uint8_2x = 28,

    automatic = 0xfe, // Not a valid HW enum value, but useful to have it here for SW
    Invalid   = 0xff  // Not a valid HW enum value, but useful to have it here for SW
};

typedef struct
{
    unsigned l1_addr_16B          : 20 __attribute__((packed));
    unsigned format               : 8 __attribute__((packed));
    unsigned /* Padding to 32b */ : 4 __attribute__((packed));
    unsigned lmt_addr_16B         : 20 __attribute__((packed));
    unsigned x_dim                : 8 __attribute__((packed));
    unsigned /* Padding to 32b */ : 4 __attribute__((packed));
    unsigned y_dim                : 8 __attribute__((packed));
    unsigned z_dim                : 8 __attribute__((packed));
    unsigned /* Padding to 32b */ : 16 __attribute__((packed));
    unsigned /* Padding to 32b */ : 32 __attribute__((packed));
} buffer_descriptor_t;

static_assert(sizeof(buffer_descriptor_t) == 16, "buffer_desc must be 128b!");

typedef union
{
    uint32_t words[4];
    buffer_descriptor_t f;
} buffer_descriptor_u;

typedef struct
{
    buffer_descriptor_u buf_desc;
    uint32_t buf_desc_id;
    unsigned reg_data_format;
} tdma_descriptor_t;

#endif
