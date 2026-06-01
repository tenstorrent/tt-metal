// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK companion to api/debug/dprint_tensix.h: a dprint_tensix_dest_reg that
// works under DstSync::SyncHalf (Metal's halts on semaphore::MATH_PACK,
// which deadlocks here) and configures the BH dest aperture format
// registers that the LLK datacopy pipeline omits. Quasar unsupported.

#pragma once

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)

#include <cstdint>

#include "api/debug/dprint_tensix.h"
#include "cfg_defines.h"
#include "dprint.h"

namespace llk_dprint
{

inline void dprint_tensix_dest_reg(int tile_id = 0)
{
    tensix_sync();

    uint32_t data_format = READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc);
    if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled))
    {
        data_format = static_cast<uint32_t>(DataFormat::Float32);
#if defined(ARCH_WORMHOLE)
        DEVICE_PRINT("WARNING: Float32 on Wormhole displays limited precision (lower 16 mantissa bits are not shown)");
#endif
    }

    DEVICE_PRINT("Tile ID = {}", tile_id);

#ifdef ARCH_BLACKHOLE
    // The BH RISC dest aperture's per-thread SEC1 registers are normally set
    // by Metal's compute API but skipped by the LLK datacopy path; do it
    // once here for the formats that read through the aperture.
    if (data_format == static_cast<uint32_t>(DataFormat::Float32) || data_format == static_cast<uint32_t>(DataFormat::Int32))
    {
        set_dest_fmt<MathThreadId>(data_format == static_cast<uint32_t>(DataFormat::Float32) ? RISC_DEST_FMT_FP32 : RISC_DEST_FMT_INT32);
        set_dest_enable_swizzling<MathThreadId>(true);
        set_dest_int8_int16_signed<MathThreadId>(false);
        tensix_sync();
    }
#endif

    uint32_t row = tile_id * NUM_ROWS_PER_TILE;
    for (uint32_t r = 0; r < NUM_ROWS_PER_TILE; ++r, ++row)
    {
        switch (data_format)
        {
            case static_cast<uint32_t>(DataFormat::Float32):
            {
                uint32_t rd[16];
#ifdef ARCH_BLACKHOLE
                const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
                for (int i = 0; i < 16; ++i)
                {
                    rd[i] = addr[i + (row << 4)];
                }
#else
                const uint16_t dr = get_dest_row_id(row, true);
                uint32_t tmp[16];
                dbg_get_array_row(dbg_array_id::DEST, dr, tmp);
                dbg_get_array_row(dbg_array_id::DEST, dr + 8, tmp + 8);
                for (int i = 0; i < 8; ++i)
                {
                    rd[2 * i]     = reconstruct_float32(lo_word(tmp[i]), lo_word(tmp[i + 8]));
                    rd[2 * i + 1] = reconstruct_float32(hi_word(tmp[i]), hi_word(tmp[i + 8]));
                }
#endif
                DEVICE_PRINT("{}", dp_typed_array_t<16>(static_cast<uint16_t>(DataFormat::Float32), rd));
                break;
            }
            case static_cast<uint32_t>(DataFormat::Int32):
            {
#ifdef ARCH_BLACKHOLE
                uint32_t rd[16];
                const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
                for (int i = 0; i < 16; ++i)
                {
                    rd[i] = addr[i + (row << 4)];
                }
                DEVICE_PRINT("{}", dp_typed_array_t<16>(static_cast<uint16_t>(DataFormat::Int32), rd));
#else
                DEVICE_PRINT("Int32 format not supported on this architecture");
#endif
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
}

} // namespace llk_dprint

#endif // defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
