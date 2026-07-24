// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Wrapper for Metal's dprint_tensix.h. We have our own dprint_tensix_dest_reg because the
// original can't run in LLK infra. It relies on dbg_halt on BH, which is a choreography across
// the TRISCs that would simply hang when run on Math, as we do here. We can't read DEST through
// the debug bus; we read RISCV_DEST_START_ADDR instead, which requires some hardware state
// programming.
//
// The Wormhole path, and Blackhole fp32/int32, are shared.
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
#ifdef ARCH_BLACKHOLE
#include "ckernel_dest.h"
#endif

inline void dprint_tensix_dest_reg(int tile_id = 0)
{
    ckernel::tensix_sync();

    DataFormat data_format = static_cast<DataFormat>(READ_HW_CFG_0_REG_FIELD(ALU_FORMAT_SPEC_REG2_Dstacc));
    if (READ_HW_CFG_0_REG_FIELD(ALU_ACC_CTRL_Fp32_enabled))
    {
        data_format = DataFormat::Float32;
#if defined(ARCH_WORMHOLE)
        DEVICE_PRINT("WARNING: Float32 on Wormhole displays limited precision - lower 16 mantissa bits are not shown\n");
#endif
    }

#if !defined(ARCH_BLACKHOLE)
    if (data_format == DataFormat::Int32)
    {
        DEVICE_PRINT("Int32 format not supported on this architecture");
        return;
    }
#endif

    DEVICE_PRINT("Tile ID = {}", tile_id);

#ifdef ARCH_BLACKHOLE
    // Program Math's SEC1 for MMIO DEST reads. No restore: this register only
    // affects the RISC MMIO window, and every MMIO consumer reprograms it.
    ckernel::configure_dest_access<ckernel::MathThreadId>(data_format, /*enable_swizzle=*/true);
    ckernel::tensix_sync();

    // Wait a bit before reading through the memory-mapped region for the write to be committed.
    ckernel::wait(100);

    constexpr uint32_t ELT_PER_ROW = 16;
    const uint32_t tile_elt_base   = tile_id * NUM_ROWS_PER_TILE * ELT_PER_ROW;
    for (uint32_t r = 0; r < NUM_ROWS_PER_TILE; ++r)
    {
        const uint32_t logical_row  = tile_id * NUM_ROWS_PER_TILE + r;
        const uint32_t row_elt_base = tile_elt_base + r * ELT_PER_ROW;
        switch (data_format)
        {
            case DataFormat::Float32:
                dprint_tensix_dest_reg_row_float32(logical_row);
                break;
            case DataFormat::Int32:
                dprint_tensix_dest_reg_row_int32(logical_row);
                break;
            case DataFormat::Float16:
            case DataFormat::Float16_b:
            case DataFormat::UInt16:
            {
                uint32_t rd[8];
                volatile uint16_t* addr = reinterpret_cast<volatile uint16_t*>(RISCV_DEST_START_ADDR);
                for (int i = 0; i < 8; ++i)
                {
                    const uint32_t lo = addr[row_elt_base + 2 * i];
                    const uint32_t hi = addr[row_elt_base + 2 * i + 1];
                    rd[i]             = lo | (hi << 16);
                }
                DEVICE_PRINT("{}", dp_typed_array_t<8>(static_cast<uint16_t>(data_format), rd));
                break;
            }
            case DataFormat::Int8:
            case DataFormat::UInt8:
            {
                uint32_t rd[4];
                volatile uint8_t* addr = reinterpret_cast<volatile uint8_t*>(RISCV_DEST_START_ADDR);
                for (int i = 0; i < 16; ++i)
                {
                    reinterpret_cast<uint8_t*>(rd)[i] = addr[row_elt_base + i];
                }
                DEVICE_PRINT("{}", dp_typed_array_t<4>(static_cast<uint16_t>(data_format), rd));
                break;
            }
            default:
                DEVICE_PRINT("Unsupported data format: {}", static_cast<uint32_t>(data_format));
                break;
        }
    }
#else
    // Wormhole (reuse Metal helpers)
    uint32_t row = tile_id * NUM_ROWS_PER_TILE;
    for (uint32_t r = 0; r < NUM_ROWS_PER_TILE; ++r, ++row)
    {
        switch (data_format)
        {
            case DataFormat::Float32:
                dprint_tensix_dest_reg_row_float32(row);
                break;
            case DataFormat::UInt16:
                dprint_tensix_dest_reg_row_uint16(static_cast<uint32_t>(data_format), row);
                break;
            case DataFormat::Float16:
            case DataFormat::Float16_b:
                dprint_tensix_dest_reg_row_float16(static_cast<uint32_t>(data_format), row);
                break;
            case DataFormat::UInt8:
                dprint_tensix_dest_reg_row_uint8(static_cast<uint32_t>(data_format), row);
                break;
            case DataFormat::Int8:
                dprint_tensix_dest_reg_row_int8(static_cast<uint32_t>(data_format), row);
                break;
            default:
                DEVICE_PRINT("Unsupported data format: {}", static_cast<uint32_t>(data_format));
                break;
        }
    }
#endif
}

#endif // defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
