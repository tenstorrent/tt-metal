// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for Metal's dprint_tensix.h.
// LLK needs its own dprint_tensix_dest_reg. It diverges from Metal in two ways:
//   - Skips dbg_halt<MathThreadId>, as it would hang.
//   - On Blackhole, reads DEST through 0xFFBD8000 for all formats. Metal does that
//     only for fp32/int32 and reads with dbg_read_dest_acc_row for other formats,
//     but that is unreliable on BH from here.
//
// Quasar is currently unsupported here; Metal device print doesn't support arrays for Quasar.

#pragma once

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)

#include <cstdint>

// Metal dprint_tensix.h needs DPRINT defined.
#include "dprint.h"
#define DPRINT(fmt, ...) DEVICE_PRINT(fmt, ##__VA_ARGS__)

#include "api/debug/dprint_tensix.h"
#include "cfg_defines.h"

inline void dprint_tensix_dest_reg(int tile_id = 0)
{
    tensix_sync();

    uint32_t data_format = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);
    if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled))
    {
        data_format = static_cast<uint32_t>(DataFormat::Float32);
#if defined(ARCH_WORMHOLE)
        DEVICE_PRINT("WARNING: Float32 on Wormhole omits lower 16 bits of DEST");
#endif
    }

#if !defined(ARCH_BLACKHOLE)
    if (data_format == static_cast<uint32_t>(DataFormat::Int32))
    {
        DEVICE_PRINT("Int32 format not supported on this architecture");
        return;
    }
#endif

    DEVICE_PRINT("Tile ID = {}", tile_id);

#ifdef ARCH_BLACKHOLE
    // Program SEC1 once up front, then read every row from 0xFFBD8000.
    // Element pointer type matches the format's element width.
    // Informed by tt_llk_blackhole/common/inc/ckernel_debug.h:dbg_copy_dest_tile.
    {
        const DataFormat fmt = static_cast<DataFormat>(data_format);
        set_dest_fmt<MathThreadId>(fmt_to_dest_type(fmt));
        set_dest_enable_swizzling<MathThreadId>(true);
        const bool is_signed = (fmt != DataFormat::UInt8) && (fmt != DataFormat::UInt16) && (fmt != DataFormat::UInt32);
        set_dest_int8_int16_signed<MathThreadId>(is_signed);
        tensix_sync();
    }

    constexpr uint32_t ELT_PER_ROW = 16;
    const uint32_t tile_elt_base   = tile_id * NUM_ROWS_PER_TILE * ELT_PER_ROW;
    for (uint32_t r = 0; r < NUM_ROWS_PER_TILE; ++r)
    {
        const uint32_t row_elt_base = tile_elt_base + r * ELT_PER_ROW;
        switch (data_format)
        {
            case static_cast<uint32_t>(DataFormat::Float32):
            case static_cast<uint32_t>(DataFormat::Int32):
            case static_cast<uint32_t>(DataFormat::UInt32):
            {
                uint32_t rd[16];
                volatile uint32_t* addr = reinterpret_cast<volatile uint32_t*>(0xFFBD8000);
                for (int i = 0; i < 16; ++i)
                {
                    rd[i] = addr[row_elt_base + i];
                }
                DEVICE_PRINT("{}", dp_typed_array_t<16>(static_cast<uint16_t>(data_format), rd));
                break;
            }
            case static_cast<uint32_t>(DataFormat::Float16):
            case static_cast<uint32_t>(DataFormat::Float16_b):
            case static_cast<uint32_t>(DataFormat::UInt16):
            {
                uint32_t rd[8];
                volatile uint16_t* addr = reinterpret_cast<volatile uint16_t*>(0xFFBD8000);
                for (int i = 0; i < 16; ++i)
                {
                    reinterpret_cast<uint16_t*>(rd)[i] = addr[row_elt_base + i];
                }
                DEVICE_PRINT("{}", dp_typed_array_t<8>(static_cast<uint16_t>(data_format), rd));
                break;
            }
            case static_cast<uint32_t>(DataFormat::Int8):
            case static_cast<uint32_t>(DataFormat::UInt8):
            {
                uint32_t rd[4]         = {0};
                volatile uint8_t* addr = reinterpret_cast<volatile uint8_t*>(0xFFBD8000);
                for (int i = 0; i < 16; ++i)
                {
                    reinterpret_cast<uint8_t*>(rd)[i] = addr[row_elt_base + i];
                }
                DEVICE_PRINT("{}", dp_typed_array_t<4>(static_cast<uint16_t>(data_format), rd));
                break;
            }
            default:
                DEVICE_PRINT("Unsupported data format: {}", data_format);
                break;
        }
    }
#else
    uint32_t row = tile_id * NUM_ROWS_PER_TILE;
    for (uint32_t r = 0; r < NUM_ROWS_PER_TILE; ++r, ++row)
    {
        switch (data_format)
        {
            case static_cast<uint32_t>(DataFormat::Float32):
            {
                const uint16_t dr = get_dest_row_id(row, true);
                uint32_t tmp[16];
                uint32_t rd[16];
                dbg_get_array_row(dbg_array_id::DEST, dr, tmp);
                dbg_get_array_row(dbg_array_id::DEST, dr + 8, tmp + 8);
                for (int i = 0; i < 8; ++i)
                {
                    rd[2 * i]     = reconstruct_float32(lo_word(tmp[i]), lo_word(tmp[i + 8]));
                    rd[2 * i + 1] = reconstruct_float32(hi_word(tmp[i]), hi_word(tmp[i + 8]));
                }
                DEVICE_PRINT("{}", dp_typed_array_t<16>(static_cast<uint16_t>(DataFormat::Float32), rd));
                break;
            }
            case static_cast<uint32_t>(DataFormat::UInt16):
            case static_cast<uint32_t>(DataFormat::Float16_b):
            case static_cast<uint32_t>(DataFormat::UInt8):
            case static_cast<uint32_t>(DataFormat::Int8):
            {
                uint32_t rd[8];
                dbg_get_array_row(dbg_array_id::DEST, get_dest_row_id(row, false), rd);
                DEVICE_PRINT("{}", dp_typed_array_t<8>(static_cast<uint16_t>(data_format), rd));
                break;
            }
            default:
                DEVICE_PRINT("Unsupported data format: {}", data_format);
                break;
        }
    }
#endif
}

#endif // defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
