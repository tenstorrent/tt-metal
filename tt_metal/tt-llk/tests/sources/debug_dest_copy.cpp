// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_dest.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif

#ifdef LLK_TRISC_MATH

using namespace ckernel;

namespace
{
constexpr std::uint32_t risc_dest_bytes_per_elem(std::uint32_t data_format)
{
    switch (data_format)
    {
        case to_underlying(DataFormat::Float32):
        case to_underlying(DataFormat::Int32):
        case to_underlying(DataFormat::UInt32):
            return 4;
        case to_underlying(DataFormat::Float16):
        case to_underlying(DataFormat::Float16_b):
        case to_underlying(DataFormat::UInt16):
            return 2;
        case to_underlying(DataFormat::Int8):
        case to_underlying(DataFormat::UInt8):
            return 1;
        default:
            return 2;
    }
}

constexpr bool risc_dest_is_unsigned(std::uint32_t data_format)
{
    switch (data_format)
    {
        case to_underlying(DataFormat::UInt8):
        case to_underlying(DataFormat::UInt16):
        case to_underlying(DataFormat::UInt32):
            return true;
        default:
            return false;
    }
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr std::uint32_t TILE_ELEMENTS = TILE_HEIGHT * TILE_WIDTH;

    const std::uint32_t l1_fmt         = formats.unpack_A_src;
    const std::uint32_t bytes_per_elem = risc_dest_bytes_per_elem(l1_fmt);
    const bool is_unsigned_int         = risc_dest_is_unsigned(l1_fmt);

    set_dest_fmt<MathThreadId>(l1_fmt);
    set_dest_enable_swizzling<MathThreadId, true>();
    set_dest_int8_int16_signed<MathThreadId>(!is_unsigned_int);
    tensix_sync();

    if (bytes_per_elem == 4)
    {
        volatile std::uint32_t* src_l1 = reinterpret_cast<volatile std::uint32_t*>(params.buffer_A[0]);
        volatile std::uint32_t* dst_l1 = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]);
        volatile std::uint32_t* dest32 = reinterpret_cast<volatile std::uint32_t*>(RISCV_DEST_START_ADDR);

        for (std::uint32_t i = 0; i < TILE_ELEMENTS; ++i)
        {
            dest32[i] = src_l1[i];
        }
        tensix_sync();
        for (std::uint32_t i = 0; i < TILE_ELEMENTS; ++i)
        {
            dst_l1[i] = dest32[i];
        }
    }
    else if (bytes_per_elem == 2)
    {
        volatile std::uint16_t* src_l1 = reinterpret_cast<volatile std::uint16_t*>(params.buffer_A[0]);
        volatile std::uint16_t* dst_l1 = reinterpret_cast<volatile std::uint16_t*>(params.buffer_Res[0]);
        volatile std::uint16_t* dest16 = reinterpret_cast<volatile std::uint16_t*>(RISCV_DEST_START_ADDR);

        for (std::uint32_t i = 0; i < TILE_ELEMENTS; ++i)
        {
            dest16[i] = src_l1[i];
        }
        tensix_sync();
        for (std::uint32_t i = 0; i < TILE_ELEMENTS; ++i)
        {
            dst_l1[i] = dest16[i];
        }
    }
    else
    {
        volatile std::uint8_t* src_l1 = reinterpret_cast<volatile std::uint8_t*>(params.buffer_A[0]);
        volatile std::uint8_t* dst_l1 = reinterpret_cast<volatile std::uint8_t*>(params.buffer_Res[0]);
        volatile std::uint8_t* dest8  = reinterpret_cast<volatile std::uint8_t*>(RISCV_DEST_START_ADDR);

        for (std::uint32_t i = 0; i < TILE_ELEMENTS; ++i)
        {
            dest8[i] = src_l1[i];
        }
        tensix_sync();
        for (std::uint32_t i = 0; i < TILE_ELEMENTS; ++i)
        {
            dst_l1[i] = dest8[i];
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
