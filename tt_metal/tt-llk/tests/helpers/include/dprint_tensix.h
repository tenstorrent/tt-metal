// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for dprint_tensix.h.
// LLK needs its own dprint_tensix_dest_reg. It diverges from Metal in two ways:
//   - Skips dbg_halt<MathThreadId>, as it would deadlock.
//   - On Blackhole, programs RISC_DEST_ACCESS_CTRL_SEC1 before reading DEST as
//     Float32 or Int32, so that tests don't have the burden of doing that themselves.
//
// Quasar is currently unsupported; Metal device print doesn't support arrays for Quasar.

#pragma once

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)

#include <cstdint>

#include "api/debug/dprint_tensix.h"
#include "cfg_defines.h"
#include "dprint.h"

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
    // Blackhole returns garbage for Float32 and Int32 through dbg_get_array_row,
    // so we read these via the memory-mapped DEST window at 0xFFBD8000.
    // Informed by tt_llk_blackhole/common/inc/ckernel_debug.h:dbg_copy_dest_tile.
    if (data_format == static_cast<uint32_t>(DataFormat::Float32) || data_format == static_cast<uint32_t>(DataFormat::Int32))
    {
        const DataFormat fmt = static_cast<DataFormat>(data_format);
        set_dest_fmt<MathThreadId>(fmt_to_dest_type(fmt));
        set_dest_enable_swizzling<MathThreadId>(true);
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
#ifdef ARCH_BLACKHOLE
            case static_cast<uint32_t>(DataFormat::Int32):
            {
                uint32_t rd[16];
                const uint32_t* addr = reinterpret_cast<const uint32_t*>(0xFFBD8000);
                for (int i = 0; i < 16; ++i)
                {
                    rd[i] = addr[i + (row << 4)];
                }

                DEVICE_PRINT("{}", dp_typed_array_t<16>(static_cast<uint16_t>(DataFormat::Int32), rd));
                break;
            }
#endif
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

#endif // defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
