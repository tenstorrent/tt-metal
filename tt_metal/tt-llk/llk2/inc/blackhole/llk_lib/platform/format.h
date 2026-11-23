// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "tensix_types.h"

// What the format fields of the unpackers, the packer and the ALU take, and the format classes their config depends on.
namespace llk2::platform::format
{

/** @brief Return the id a THCON or ALU format field takes for a DataFormat: its value without the high flag bits. */
constexpr std::uint32_t id(const DataFormat format)
{
    return ckernel::masked_data_format(static_cast<std::uint32_t>(format));
}

/** @brief Return the bytes one datum of a format takes in Dest or in the packer's input. */
constexpr std::uint32_t datum_bytes(const DataFormat format)
{
    return ckernel::datum_size_in_bytes(id(format));
}

/** @brief Return whether a format has a 5-bit exponent (the A formats: Float16, Bfp8, Bfp4, Bfp2, Lf8). */
constexpr bool has_5bit_exponent(const DataFormat format)
{
    return ckernel::IS_A_FORMAT(id(format));
}

/** @brief Return whether a format is a block float with a 5-bit exponent (Bfp8, Bfp4, Bfp2). */
constexpr bool is_block_float_5bit_exponent(const DataFormat format)
{
    return ckernel::IS_BFP_A_FORMAT(id(format));
}

/** @brief Return whether the ALU runs a format in its integer mode (INT8_math_enabled). */
constexpr bool uses_integer_math(const DataFormat format)
{
    return ckernel::is_int8_or_int32_format(static_cast<std::uint32_t>(format));
}

} // namespace llk2::platform::format
