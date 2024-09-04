// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSIX_TYPES_H_INCLUDED
#define TENSIX_TYPES_H_INCLUDED

#include <cstdint>

#ifndef TENSIX_FIRMWARE
#include <fmt/core.h>
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
typedef enum {
    XMOV_L0_TO_L1 = 0,
    XMOV_L1_TO_L0 = 1,
    XMOV_L0_TO_L0 = 2,
    XMOV_L1_TO_L1 = 3,
} xmov_direction_t;

typedef enum { TDMA_MOVER0 = 0, TDMA_MOVER1 = 1 } tdma_mover_id_t;

typedef enum { MATH_HF = 1, MATH_AUTO = 2, MATH_LF = 4 } math_fidelity_t;

typedef enum { RELU_NONE = 0, RELU_PLAIN = 1, RELU_THRESH = 2, RELU_MAX = 3 } relu_mode_t;

/////////////
// TDMA Registers
////////////
typedef struct {
    uint32_t row_section_size : 16;
    uint32_t exp_section_size : 16;
    uint32_t tile_dst_addr : 32;
    uint32_t uncompressed : 1;
    uint32_t reserved_0 : 3;
    uint32_t out_data_format : 2;
    uint32_t reserved_1 : 2;
    uint32_t in_data_format : 2;
    uint32_t reserved_2 : 22;
    uint32_t reserved_3 : 32;
} packer_config_t;  // 16B

struct fifo_ctl_t {
    uint32_t rd_ptr;
    uint32_t wr_ptr;
    uint32_t rsvd0;
    uint32_t rsvd1;
#ifndef TENSIX_FIRMWARE
    operator std::string() const {
        return fmt::format("Fifo Control: rd_ptr(0x{:08x}) wr_ptr(0x{:08x})", rd_ptr, wr_ptr);
    }
#endif
};

typedef struct {
    uint32_t val[4];
    packer_config_t f;
} packer_config_u;

typedef struct {
    uint32_t src_addr : 32;
    uint32_t dst_addr : 32;
    uint32_t xfer_size : 32;
    uint32_t xfer_dir : 2;
    uint32_t reserved_0 : 30;
} mover_config_t;  // 16B

typedef struct {
    uint32_t val[4];
    mover_config_t f;
} mover_config_u;

/////////////
// Data section structures
/////////////

// Tile descriptor
typedef struct {
    uint32_t data_format : 4;
    uint32_t uncompressed : 1;
    uint32_t reserved_0 : 3;
    uint32_t blobs_per_xy_plane : 4;
    uint32_t reserved_1 : 4;
    uint32_t x_dim : 16;
    uint32_t y_dim : 16;
    uint32_t z_dim : 16;
    uint32_t w_dim : 16;
    uint32_t blobs_y_start : 32;
    uint32_t digest_type : 8;  // Not used
    uint32_t digest_size : 8;  // Not used
} tile_descriptor_t;           // Unpack configuration

typedef union {
    uint32_t val[4];
    tile_descriptor_t f;
} tile_descriptor_u;

struct TileHeader {
    // occupied part of the 16B line
    std::uint16_t tile_size_16B = 0;
    std::uint16_t reserved_0_mbz : 1;
    std::uint16_t tile_id : 15;

    std::uint8_t metadata_size_16B = 0;
    std::uint8_t reserved_1 = 0;
    std::uint16_t format = 0x10;  // [3:0] format, 4-uncompress flag.

    std::uint32_t zero_mask = 0;
    std::uint32_t reserved_3 = 0;

    TileHeader() : reserved_0_mbz(0), tile_id(0) {}

    bool IsCompressed() const { return ((format & 0x10) == 0); }

#ifndef TENSIX_FIRMWARE
    operator std::string() const {
        return fmt::format("TileHeader:tile_id(0x{:04x}) size16B(0x{:04x})", tile_id, tile_size_16B);
    }

    std::size_t size() const { return 16; }
    const void *data() const { return this; }
    typedef std::uint8_t value_type;

    bool operator!=(const TileHeader &rhs) const {
        bool result =
            tile_size_16B != rhs.tile_size_16B || tile_id != rhs.tile_id || metadata_size_16B != rhs.metadata_size_16B;
        return result;
    }

#endif
};

union TileHeader_u {
    uint32_t val[4];
    TileHeader header;
    TileHeader_u(){};
};

static_assert(sizeof(TileHeader) == 16, "TileHeader must be 16B");

struct SectionHeader {
    // occupied part of the 16B line
    std::uint16_t section_id;
    std::uint16_t section_size;
    std::uint16_t tile_count;

    // unoccupied part of the 16B line
    std::uint16_t reserved[5];

#ifndef TENSIX_FIRMWARE
    operator std::string() const {
        return fmt::format(
            "SectionHeader: id(0x{:04x}) size(0x{:04x}) tile_count(0x{:04x})", section_id, section_size, tile_count);
    }
#endif
};

// Actually it only has to be a multiple of 16B
static_assert(sizeof(SectionHeader) == 16, "struct section_header must be 16 bytes");

static constexpr std::uint32_t TEST_MSG_EN_TENSIX_PM = 0;
static constexpr std::uint32_t TEST_MSG_DBG_DISABLE = 1;
static constexpr std::uint32_t TEST_MSG_SET_MAX_EXP_THRESH = 2;
static constexpr std::uint32_t TEST_MSG_RISC_BP_DISABLE = 3;
static constexpr std::uint32_t TEST_MSG_SET_RELU_PARAMS = 4;
static constexpr std::uint32_t TEST_MSG_SYNTH_CKERNEL = 10;

static constexpr std::uint32_t COMMAND_QUEUE_SIZE_BYTES_LOG2 = 16;
static constexpr std::uint32_t COMMAND_QUEUE_SIZE_BYTES = 1 << COMMAND_QUEUE_SIZE_BYTES_LOG2;
static constexpr std::uint32_t COMMAND_SIZE_BYTES_LOG2 = 5;
static constexpr std::uint32_t COMMAND_SIZE_BYTES = 1 << COMMAND_SIZE_BYTES_LOG2;

static constexpr std::uint32_t DEST_FACE_WIDTH = 16;
static constexpr std::uint32_t DEST_FACE_WIDTH_LOG2 = 4;
static constexpr std::uint32_t DEST_FACE_HEIGHT = 16;
static constexpr std::uint32_t DEST_FACE_HEIGHT_LOG2 = 4;
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE = 64 * DEST_FACE_HEIGHT;
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE_LOG2 = 10;
static constexpr std::uint32_t DEST_REGISTER_HALF_SIZE = DEST_REGISTER_FULL_SIZE / 2;

#ifdef TENSIX_FIRMWARE
static constexpr std::uint32_t L1_SIZE = 1 << 20;
#else
static constexpr std::uint32_t L1_SIZE = 1 << 21;
#endif

// Voluntary FIFO alignment so that we can pack fifo address down to 16 bits in the command.
// At 8, we can cover 16MB. The upper limit is 15 because the command queue is at 32K (by default)
// Even though the command queue address never goes into a command, assertions complain if it is misaligned.
// (Hardware only requires 16B alignment.)
static constexpr std::uint32_t FIFO_BASE_ADDRESS_ALIGN_BITS = 9;
static constexpr std::uint32_t FIFO_BASE_ADDRESS_ALIGN = 1 << FIFO_BASE_ADDRESS_ALIGN_BITS;

enum class DataFormat : std::uint8_t {
    Float32 = 0,
    Float16 = 1,
    Bfp8 = 2,
    Bfp4 = 3,
    Bfp2 = 11,
    Float16_b = 5,
    Bfp8_b = 6,
    Bfp4_b = 7,
    Bfp2_b = 15,
    Lf8 = 10,
    Int8 = 14,
    UInt8 = 30,
    UInt16 = 9,
    Int32 = 8,
    UInt32 = 24,
    Tf32 = 0xff,
    Invalid = 0xff
};

struct io_queue_pointers_t {
    static constexpr std::uint32_t INVALID_IO_QUEUE_POINTER = 0xfeedface;
    static constexpr std::uint32_t WRAP_MASK = 0x80000000;
    static constexpr std::uint32_t MAX_IO_QUEUES = 256;
    static constexpr std::uint32_t INPUT_IO_QUEUES = 64;

    std::uint32_t rdptr;
    std::uint32_t wrptr;
    std::uint32_t base_addr;
    std::uint32_t data_size_16B;
    std::uint32_t buffer_size_16B;

    inline void init_input_queue(
        std::uint32_t buffer_start, std::uint32_t buffer_end, std::uint32_t data_size) volatile {
        base_addr = buffer_start;
        rdptr = buffer_start;
        data_size_16B = data_size >> 4;
        buffer_size_16B = (buffer_end - buffer_start) >> 4;
    }

    inline void init_output_queue(
        std::uint32_t buffer_start, std::uint32_t buffer_end, std::uint32_t data_size) volatile {
        base_addr = buffer_start;
        wrptr = buffer_start;
        data_size_16B = data_size >> 4;
        buffer_size_16B = (buffer_end - buffer_start) >> 4;
    }

    inline void reset() volatile {
        rdptr = INVALID_IO_QUEUE_POINTER;
        wrptr = INVALID_IO_QUEUE_POINTER;
    }

    inline bool valid() volatile { return (rdptr != INVALID_IO_QUEUE_POINTER); }

    inline std::uint32_t get_buffer_end() const volatile { return base_addr + (buffer_size_16B << 4); }

    inline void increment_rd_pointer() volatile {
        if (!valid())
            return;
        std::uint32_t new_rdptr = rdptr + (data_size_16B << 4);
        if ((new_rdptr & ~WRAP_MASK) >= get_buffer_end()) {
            if (wrap_bit(new_rdptr)) {
                new_rdptr = base_addr;
            } else {
                new_rdptr = WRAP_MASK | base_addr;
            }
        }
        rdptr = new_rdptr;
    }

    inline bool wrap_bit(std::uint32_t ptr) volatile { return (ptr & WRAP_MASK) != 0; }

    inline void increment_wr_pointer() volatile {
        if (wrptr == INVALID_IO_QUEUE_POINTER)
            return;
        std::uint32_t new_wrptr = wrptr + (data_size_16B << 4);
        if ((new_wrptr & ~WRAP_MASK) >= get_buffer_end()) {
            if (wrap_bit(new_wrptr)) {
                new_wrptr = base_addr;
            } else {
                new_wrptr = WRAP_MASK | base_addr;
            }
        }
        wrptr = new_wrptr;
    }

    inline void set_wr_pointer(std::uint32_t value) volatile { wrptr = value; }

    inline void set_rd_pointer(std::uint32_t value) volatile { rdptr = value; }

    inline bool empty() volatile { return rdptr == wrptr; }

    inline bool full() volatile {
        auto wrapped_rdptr = rdptr ^ WRAP_MASK;
        return wrapped_rdptr == wrptr;
    }

    inline bool has_data() volatile {
        return (rdptr != INVALID_IO_QUEUE_POINTER) and (wrptr != INVALID_IO_QUEUE_POINTER) and (not empty());
    }

    inline std::uint32_t unwrap_ptr(std::uint32_t value) const volatile {
        if (value == INVALID_IO_QUEUE_POINTER) {
            return value;
        }
        return value & ~WRAP_MASK;
    }
};

#endif
