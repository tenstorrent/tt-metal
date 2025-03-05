// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdarg>
#include <type_traits>

#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_square.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "tensix_types.h"

inline uint32_t L1_ADDRESS(const volatile void* buffer)
{
    return (reinterpret_cast<uint32_t>(buffer) / 16) - 1;
}

namespace
{
constexpr std::underlying_type_t<DataFormat> get_data_format(DataFormat format)
{
    return static_cast<std::underlying_type_t<DataFormat>>(format);
}
} // namespace

#ifdef FORMAT_FLOAT16_B
constexpr auto DATA_FORMAT = get_data_format(DataFormat::Float16_b);
#endif
#ifdef FORMAT_FLOAT16
constexpr auto DATA_FORMAT = get_data_format(DataFormat::Float16);
#endif
#ifdef FORMAT_FLOAT32
constexpr auto DATA_FORMAT = get_data_format(DataFormat::Float32);
#endif
#ifdef FORMAT_INT32
constexpr auto DATA_FORMAT = get_data_format(DataFormat::Int32);
#endif
#ifdef FORMAT_BFP8_B
constexpr auto DATA_FORMAT = get_data_format(DataFormat::Bfp8_b);
#endif

#ifdef ELTWISE_BINARY_ADD
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWADD;
#endif
#ifdef ELTWISE_BINARY_SUB
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWSUB;
#endif
#ifdef ELTWISE_BINARY_MUL
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;
#endif
// TO BE IMPLEMENTED IN LLKs
#ifdef ELTWISE_BINARY_DIV
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWDIV;
#endif
#ifdef ELTWISE_BINARY_LESS
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWLESS;
#endif

#ifdef SFPU_OP_SQRT
constexpr auto SFPU_OPERATION = SfpuType::sqrt;
#endif
#ifdef SFPU_OP_LOG
constexpr auto SFPU_OPERATION = SfpuType::log;
#endif
#ifdef SFPU_OP_SQUARE
constexpr auto SFPU_OPERATION = SfpuType::square;
#endif

inline void process_addresses(volatile uint32_t* buffer_Dest[], int n, int first, ...)
{
    buffer_Dest[0] = reinterpret_cast<volatile uint32_t*>(first);

    va_list args;
    va_start(args, first);
    for (int i = 1; i < n; ++i)
    {
        int num        = va_arg(args, int);
        buffer_Dest[i] = reinterpret_cast<volatile uint32_t*>(num);
    }
    va_end(args);
}
