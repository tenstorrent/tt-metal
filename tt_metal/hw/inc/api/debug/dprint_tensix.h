// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_debug.h"
#if !defined(ENV_LLK_INFRA)
#include "api/compute/compute_kernel_api.h"
#include "dprint.h"
#else
inline void dbg_read_dest_acc_row(int row_addr, uint32_t* rd_data) {
    ckernel::dbg_get_array_row(ckernel::dbg_array_id::DEST, row_addr, rd_data);
}
#endif
#include "tensix_types.h"

// Given a Tensix configuration register field name, print the contents of the register.
// Uses tt_metal/hw/inc/<family>/cfg_defines.h:
//   For config section "Registers for THREAD", use banks THREAD_0_CFG, THREAD_1_CFG, THREAD_2_CFG
//   For other config sections (ALU,PACK0), use banks HW_CFG_0, HW_CFG_1
#define READ_CFG_REG_FIELD(bank, reg_field_name) \
    (ckernel::dbg_read_cfgreg(bank, reg_field_name##_ADDR32) & reg_field_name##_MASK) >> reg_field_name##_SHAMT

// Helper macros
#define READ_HW_CFG_0_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_0, reg_field_name)
#define READ_HW_CFG_1_REG_FIELD(reg_field_name) READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::HW_CFG_1, reg_field_name)
#define READ_THREAD_0_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_0_CFG, reg_field_name)
#define READ_THREAD_1_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_1_CFG, reg_field_name)
#define READ_THREAD_2_CFG_REG_FIELD(reg_field_name) \
    READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::THREAD_2_CFG, reg_field_name)

constexpr int PRECISION = 4;
constexpr int WIDTH = 8;

constexpr uint16_t NUM_FACES_PER_TILE = 4;
constexpr uint16_t NUM_ROWS_PER_FACE = 16;
constexpr uint16_t NUM_ROWS_PER_TILE = NUM_FACES_PER_TILE * NUM_ROWS_PER_FACE;

// Helper function to print array
template <uint32_t count>
inline void dprint_array_with_data_type(uint32_t data_format, uint32_t* data) {
    DPRINT("{}\n", dp_typed_array_t<count>(data_format, data));
}

#ifndef ARCH_QUASAR  // references Bfp*/Lf8/UInt32 formats absent from the Quasar DataFormat enum
// Dprints data format as string given an uint
inline void dprint_data_format(uint8_t data_format) {
    switch (data_format) {
        case (uint8_t)DataFormat::Float32: DPRINT("Float32"); break;
        case (uint8_t)DataFormat::Float16: DPRINT("Float16"); break;
        case (uint8_t)DataFormat::Bfp8: DPRINT("Bfp8"); break;
        case (uint8_t)DataFormat::Bfp4: DPRINT("Bfp4"); break;
        case (uint8_t)DataFormat::Bfp2: DPRINT("Bfp2"); break;
        case (uint8_t)DataFormat::Float16_b: DPRINT("Float16_b"); break;
        case (uint8_t)DataFormat::Bfp8_b: DPRINT("Bfp8_b"); break;
        case (uint8_t)DataFormat::Bfp4_b: DPRINT("Bfp4_b"); break;
        case (uint8_t)DataFormat::Bfp2_b: DPRINT("Bfp2_b"); break;
        case (uint8_t)DataFormat::Lf8: DPRINT("Lf8"); break;
        case (uint8_t)DataFormat::Int8: DPRINT("Int8"); break;
        case (uint8_t)DataFormat::UInt8: DPRINT("UInt8"); break;
        case (uint8_t)DataFormat::UInt16: DPRINT("UInt16"); break;
        case (uint8_t)DataFormat::Int32: DPRINT("Int32"); break;
        case (uint8_t)DataFormat::UInt32: DPRINT("UInt32"); break;
        case (uint8_t)DataFormat::Tf32: DPRINT("Tf32"); break;
        default: DPRINT("INVALID DATA FORMAT"); break;
    }
}
#else
// Declared and deleted rather than simply absent, so that a Quasar build which does reach this
// function fails at the call site with "use of deleted function 'dprint_data_format'" instead of a
// bare "not declared in this scope". An `#else #error` cannot be used here: this header is also
// Quasar's DEST-print path, so it must keep compiling on Quasar, and an #error would fire on every
// Quasar build rather than only on an actual reference.
void dprint_data_format(uint8_t data_format) = delete;
#endif  // !ARCH_QUASAR

// if flag DEST_ACCESS_CFG_remap_addrs is enabled
// destination register row identifiers are remmaped
// bits 5:3 are rotated 543 -> 354
inline uint16_t get_remapped_row_id(uint16_t row_id) {
    // bits 5:3 are rotating -> 543 -> 354
    return (row_id & 0xFFC7) |         // clear bits [5:3]
           ((row_id & 0x0008) << 2) |  // shifting bit 3 to position 5
           ((row_id & 0x0030) >> 1);   // shifting bits 5:4 to position 4:3
}

// if flag DEST_ACCESS_CFG_swizzle_32b is enabled dest address is has bits [3:2] shuffled
inline uint16_t get_swizzled_row_id(uint16_t row_id) {
    if (row_id & 0x10) {
        switch ((row_id & 0xC) >> 2) {
            case 0: return (row_id & 0xFFF3) | 0x8;
            case 1: return (row_id & 0xFFF3);
            case 2: return (row_id & 0xFFF3) | 0xC;
            case 3:
            default: return (row_id & 0xFFF3) | 0x4;
        }
    } else {
        return (row_id & 0xFFF3) | ((row_id & 0x4) << 1) | ((row_id & 0x8) >> 1);
    }
}

inline uint16_t get_logical_row_id(uint16_t tile_id, uint16_t face_id, uint16_t row_id) {
    return NUM_ROWS_PER_TILE * tile_id + NUM_ROWS_PER_FACE * face_id + row_id;
}

// Calculates dest row address based on logical row identifiers (tile_id, face_id, row_id)
// and dest configuration.
inline uint16_t get_dest_row_id(uint16_t logical_row_id, bool is_float32) {
    uint16_t row = logical_row_id;

#ifdef ARCH_BLACKHOLE
    if (READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_remap_addrs) == 1) {
        row = get_remapped_row_id(row);
    }
#endif

    if (is_float32) {
#ifdef ARCH_BLACKHOLE
        if (READ_HW_CFG_0_REG_FIELD(DEST_ACCESS_CFG_swizzle_32b) == 1) {
            row = get_swizzled_row_id(row);
        }
#endif
        // 0-7  dest rows for Float16
        // 8-15 dest rows for Mantissa
        // need to shift row index starting from bit 3
        row = ((row & 0xFFF8) << 1) | (row & 0x7);
    }

    return row;
}

inline uint16_t lo_word(uint32_t dword) { return dword & 0xFFFF; }
inline uint16_t hi_word(uint32_t dword) { return lo_word(dword >> 16); }

// Float16 = [1-bit sign, 7-bit mantissa, 8-bit exponent]
// Mantissa16 = [16-bit mantissa]
// Float32 = [1-bit sign, 8-bit exponent, 23-bit mantissa(7-bit + 16-bit)]
inline uint32_t reconstruct_float32(uint32_t float16, uint32_t mantissa16) {
    uint32_t sign = (float16 & 0x00008000) << 16;
    uint32_t exponent = (float16 & 0x000000FF) << 23;
    uint32_t mantissa = ((float16 & 0x00007F00) << 8) | mantissa16;

    return sign | exponent | mantissa;
}

#ifndef ARCH_QUASAR  // these DEST readers use the RISCV_DEBUG_REG_* debug-bus wrapper macros, which are not wired up on
                     // Quasar
// Helper function that prints one row from dest when dest is configured for storing float32 values.
// This function should be used only from dprint_tensix_dest_reg.
// Float32 in dest = [Float16, Mantissa16]
// dest_row -> [[Float16_1,Float16_0],...[Float16_15, Float16_14]]
// dest_row + 8 -> [[Mantissa16_1,Mantissa16_0],...[Mantissa16_15, Mantissa16_14]]
inline void dprint_tensix_dest_reg_row_float32(uint16_t row) {
    constexpr int ARRAY_LEN = 16;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type

#ifdef ARCH_BLACKHOLE
    // On Blackhole, use direct dest access - Float32 values are already in correct format
    const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
    for (int i = 0; i < ARRAY_LEN; ++i) {
        rd_data[i] = addr[i + (row << 4)];
    }
#else
    // On other architectures, need to reconstruct Float32 from Float16 and Mantissa16
    row = get_dest_row_id(row, true);
    uint32_t rd_data_temp[ARRAY_LEN];
    dbg_read_dest_acc_row(row, rd_data_temp);
    dbg_read_dest_acc_row(row + 8, rd_data_temp + 8);

    for (int i = 0; i < 8; ++i) {
        rd_data[2 * i] = reconstruct_float32(lo_word(rd_data_temp[i]), lo_word(rd_data_temp[i + 8]));
        rd_data[2 * i + 1] = reconstruct_float32(hi_word(rd_data_temp[i]), hi_word(rd_data_temp[i + 8]));
    }
#endif

    dprint_array_with_data_type<ARRAY_LEN>((uint32_t)DataFormat::Float32, rd_data);
}

// Helper function that prints one row from dest when dest is configured for storing float16 values.
// This function should be used only from dprint_tensix_dest_reg.
inline void dprint_tensix_dest_reg_row_float16(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    row = get_dest_row_id(row, false);
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type<ARRAY_LEN>(data_format, rd_data);
}

inline void dprint_tensix_dest_reg_row_int32([[maybe_unused]] uint16_t row) {
#ifdef ARCH_BLACKHOLE
    constexpr int ARRAY_LEN = 16;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
    for (int i = 0; i < ARRAY_LEN; ++i) {
        rd_data[i] = addr[i + (row << 4)];
    }
    dprint_array_with_data_type<ARRAY_LEN>((uint32_t)DataFormat::Int32, rd_data);
#else
    DPRINT("Int32 format not supported on this architecture\n");
#endif
}

// Helper function that prints one row from dest when dest is configured for storing uint16 values.
// This function should be used only from dprint_tensix_dest_reg.
inline void dprint_tensix_dest_reg_row_uint16(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    row = get_dest_row_id(row, false);
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type<ARRAY_LEN>(data_format, rd_data);
}

// Helper function that prints one row from dest when dest is configured for storing uint8 values.
// This function should be used only from dprint_tensix_dest_reg.
inline void dprint_tensix_dest_reg_row_uint8(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    row = get_dest_row_id(row, false);
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type<ARRAY_LEN>(data_format, rd_data);
}

inline void dprint_tensix_dest_reg_row_int8(uint32_t data_format, uint16_t row) {
    constexpr int ARRAY_LEN = 8;
    uint32_t rd_data[ARRAY_LEN + 1];  // data + array type
    row = get_dest_row_id(row, false);
    dbg_read_dest_acc_row(row, rd_data);
    dprint_array_with_data_type<ARRAY_LEN>(data_format, rd_data);
}
#else
// Deleted on Quasar for the same reason as dprint_data_format above: these read DEST through the
// debug bus, so a reference from a Quasar build is a bug, and a deleted declaration names the
// offending call site instead of failing as an undeclared identifier.
void dprint_tensix_dest_reg_row_float32(uint16_t row) = delete;
void dprint_tensix_dest_reg_row_float16(uint32_t data_format, uint16_t row) = delete;
void dprint_tensix_dest_reg_row_int32(uint16_t row) = delete;
void dprint_tensix_dest_reg_row_uint16(uint32_t data_format, uint16_t row) = delete;
void dprint_tensix_dest_reg_row_uint8(uint32_t data_format, uint16_t row) = delete;
void dprint_tensix_dest_reg_row_int8(uint32_t data_format, uint16_t row) = delete;
#endif  // !ARCH_QUASAR

#if !defined(ENV_LLK_INFRA)
#ifdef ARCH_QUASAR
// The shared typed-array print path renders 16-bit datums with the host's make_float(), which expects
// Tensix DEST field order -- [sign][mantissa][exponent] -- the raw layout the debug-bus read returns on
// the other architectures. Quasar's memory-mapped DEST window hands back standard IEEE order instead,
// so the two fields have to be swapped back for the rendered text to agree. Float32 needs no swap: the
// host bit-casts those words straight to float.
inline uint32_t dest_order_from_ieee_float16_b(uint32_t ieee) {
    return (ieee & 0x8000u) | ((ieee & 0x7Fu) << 8) | ((ieee >> 7) & 0xFFu);
}

// Same swap for IEEE half (Float16): [sign][exp5][man10] -> [sign][man10][exp5].
inline uint32_t dest_order_from_ieee_float16(uint32_t ieee) {
    return (ieee & 0x8000u) | ((ieee & 0x3FFu) << 5) | ((ieee >> 10) & 0x1Fu);
}

// Applies a 16-bit field-order swap (one of the two above) to both datums of a word, low half first.
inline uint32_t dest_order_from_ieee_pair(uint32_t word, uint32_t (*to_dest_order)(uint32_t)) {
    return to_dest_order(word & 0xFFFFu) | (to_dest_order(word >> 16) << 16);
}

// Recovers the tile's bits for four Int8 datums (one byte each) read through the signed Int8 view.
// The unpacker decodes an Int8 tile as sign-magnitude -- bit 7 the sign, bits 6:0 the magnitude m --
// and the signed view returns that value in two's complement, so a negative datum comes back as
// 0x100 - m. Re-encoding it as 0x80 | m gives back the tile's byte, which the host then prints as two's
// complement, as on WH/BH.
inline uint32_t int8_tile_bits_from_dest_view(uint32_t word) {
    uint32_t out = 0;
    for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (word >> (8 * b)) & 0xFFu;
        const uint32_t tile_byte = (byte & 0x80u) ? 0x80u | ((0x100u - byte) & 0x7Fu) : byte;
        out |= tile_byte << (8 * b);
    }
    return out;
}

// Prints one DEST row -- 16 datums -- as a typed array: 4 words of 8-bit, 8 words of 16-bit or 16 words
// of 32-bit datums. The Math section must already be programmed for data_format's read view.
inline void dprint_tensix_dest_row(DataFormat data_format, uint32_t row) {
    if (data_format == DataFormat::Int8) {
        uint32_t rd_data[4];
        ckernel::dbg_read_dest_row_8b(row, rd_data);
        for (uint32_t& word : rd_data) {
            word = int8_tile_bits_from_dest_view(word);
        }
        dprint_array_with_data_type<4>((uint32_t)data_format, rd_data);
    } else if (data_format == DataFormat::Float32 || data_format == DataFormat::Int32) {
        uint32_t rd_data[16];
        ckernel::dbg_read_dest_row_32b(row, rd_data);
        dprint_array_with_data_type<16>((uint32_t)data_format, rd_data);
    } else {
        uint32_t rd_data[8];
        ckernel::dbg_read_dest_row_16b(row, rd_data);
        const auto to_dest_order =
            data_format == DataFormat::Float16 ? dest_order_from_ieee_float16 : dest_order_from_ieee_float16_b;
        for (uint32_t& word : rd_data) {
            word = dest_order_from_ieee_pair(word, to_dest_order);
        }
        dprint_array_with_data_type<8>((uint32_t)data_format, rd_data);
    }
}

// Prints the contents of tile tile_id within the destination register, row by row, in the same typed
// array form the other architectures emit -- the host print parser decodes it, so the rendered text is
// identical everywhere. The DEST data format is passed in rather than recovered from config, because
// the RISCV_DEBUG_REG_* config-read wrappers are not wired up on Quasar. Supports Float32, Float16,
// Float16_b, Int32 and Int8. Pass the format DEST holds, not the tile's L1 format: block formats (MxFp8R/P,
// MxFp6R/P, MxFp4, MxInt8/4/2) unpack into DEST as Float16_b, or as Float32 with a 32-bit DEST.
//
// Call this only between tile_regs_acquire() and tile_regs_commit(). The unpack<->math mailbox
// rendezvous (dbg_thread_halt) quiesces unpack for the duration of the read, so an in-flight unpack
// cannot desync the tile counter underneath it (TILE_COUNTERS fault). Pack is not a participant in
// that rendezvous, so keeping pack off DEST is the caller's responsibility, and the acquire-to-commit
// window is what provides it: math owns DEST there and pack is still waiting on it. Called outside
// that window, this races an in-flight pack.
inline void dprint_tensix_dest_reg(DataFormat data_format, int tile_id = 0) {
    UNPACK(ckernel::dbg_thread_halt<ckernel::UnpackThreadId>());
    MATH(ckernel::dbg_thread_halt<ckernel::MathThreadId>());
    MATH({
        // Reading DEST at the wrong element width yields plausible-looking garbage rather than an
        // obvious failure, so refuse formats this path has not been validated against. Note the
        // rendezvous is still entered and left symmetrically -- returning early here would strand
        // unpack in dbg_thread_halt.
        if (data_format != DataFormat::Float32 && data_format != DataFormat::Float16 &&
            data_format != DataFormat::Float16_b && data_format != DataFormat::Int32 &&
            data_format != DataFormat::Int8) {
            DPRINT(
                "dprint_tensix_dest_reg: unsupported data format {}, expected Float32, Float16, Float16_b, Int32 or "
                "Int8\n",
                (uint32_t)data_format);
        } else {
            // As on WH/BH, integers are printed as the raw bits DEST holds, rendered as two's complement. That
            // is right for tensors the host wrote and a datacopy moved into DEST; sign-magnitude results of
            // the FPU (e.g. integer matmul) print their negatives wrong, as they do on WH/BH. Int32 is read
            // through the window's Float32 leg, which returns each 32-bit word unchanged.
            const DataFormat read_format = data_format == DataFormat::Int32 ? DataFormat::Float32 : data_format;
            // Program Math's section for MMIO DEST reads. configure_dest_access issues RMWCIB config writes,
            // so wait for the config unit before the first read.
            ckernel::configure_dest_access<ckernel::MathThreadId>(read_format, /*enable_swizzle=*/true);
            ckernel::wait_cfg_idle();

            DPRINT("Tile ID = {}\n", tile_id);
            for (uint32_t i = 0; i < NUM_ROWS_PER_TILE; ++i) {
                dprint_tensix_dest_row(data_format, tile_id * NUM_ROWS_PER_TILE + i);
            }
        }
    })
    MATH(ckernel::dbg_thread_unhalt<ckernel::MathThreadId>());
}
#else
// Print the contents of tile with index tile_id within the destination register
template <bool print_by_face = false>
void dprint_tensix_dest_reg(int tile_id = 0) {
    dbg_halt();
    MATH({
        // Determine the format of the data in the destination register
        uint32_t data_format_reg_field_value = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);

        if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled)) {
            data_format_reg_field_value = (uint32_t)DataFormat::Float32;
#if defined(ARCH_WORMHOLE)
            DPRINT("WARNING: Float32 on Wormhole displays limited precision - lower 16 mantissa bits are not shown\n");
#endif
        }

        // Print the contents
        DPRINT("Tile ID = {}\n", tile_id);

        uint32_t row = tile_id * NUM_ROWS_PER_TILE;
        for (int face_id = 0; face_id < NUM_FACES_PER_TILE; ++face_id) {
            for (int row_id = 0; row_id < NUM_ROWS_PER_FACE; ++row_id) {
                switch (data_format_reg_field_value) {
                    case (uint32_t)DataFormat::Float32: dprint_tensix_dest_reg_row_float32(row); break;
                    case (uint32_t)DataFormat::Int32: dprint_tensix_dest_reg_row_int32(row); break;
                    case (uint32_t)DataFormat::UInt16:
                        dprint_tensix_dest_reg_row_uint16(data_format_reg_field_value, row);
                        break;
                    case (uint32_t)DataFormat::Float16:
                    case (uint32_t)DataFormat::Float16_b:
                        dprint_tensix_dest_reg_row_float16(data_format_reg_field_value, row);
                        break;
                    case (uint32_t)DataFormat::UInt8:
                        dprint_tensix_dest_reg_row_uint8(data_format_reg_field_value, row);
                        break;
                    case (uint32_t)DataFormat::Int8:
                        dprint_tensix_dest_reg_row_int8(data_format_reg_field_value, row);
                        break;
                    default: DPRINT("Unsupported data format: {}\n", data_format_reg_field_value); break;
                }
                row++;
            }
            if constexpr (print_by_face) {
                DPRINT("\n");
            }
        }
    })
    dbg_unhalt();
}
#endif  // ARCH_QUASAR
#endif  // !defined(ENV_LLK_INFRA)

// Print the contents of the specified configuration register field.
// Example:
//   dprint_cfg_reg_field(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg_field(bank, reg_field_name)                                          \
    {                                                                                       \
        uint32_t field_val = READ_CFG_REG_FIELD(ckernel::dbg_cfgreg::bank, reg_field_name); \
        DPRINT(#reg_field_name " = {}\n", field_val);                                       \
    }

// Print the contents of the whole configuration register. The register is specified by
// the name of any field within it.
// Example:
//    dprint_cfg_reg(HW_CFG_0,ALU_FORMAT_SPEC_REG2_Dstacc);
#define dprint_cfg_reg(bank, reg_field_name)                                                    \
    {                                                                                           \
        uint32_t reg_val = dbg_read_cfgreg(ckernel::dbg_cfgreg::bank, reg_field_name##_ADDR32); \
        DPRINT(#reg_field_name " = 0x{:x}\n", reg_val);                                         \
    }

// Print the content of the register field given the value in the register.
#define DPRINT_TENSIX_CONFIG_FIELD(reg_val, reg_field_name, name, printDec)                 \
    {                                                                                       \
        uint32_t field_value = (reg_val & reg_field_name##_MASK) >> reg_field_name##_SHAMT; \
        if (printDec) {                                                                     \
            DPRINT(name " = {}; ", field_value);                                            \
        } else {                                                                            \
            DPRINT(name " = 0x{:x}; ", field_value);                                        \
        }                                                                                   \
    }

#define dprint_tensix_struct_field(word, mask, shamt, name, printDec) \
    {                                                                 \
        if (printDec) {                                               \
            DPRINT(name ": {}\n", ((word) & (mask)) >> (shamt));      \
        } else {                                                      \
            DPRINT(name ": 0x{:x}\n", ((word) & (mask)) >> (shamt));  \
        }                                                             \
    }
