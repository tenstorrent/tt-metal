// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdarg>
#include <type_traits>

// Include auto-generated build configuration
#include "build.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "data_format_inference.h"
#include "perf.h"
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

constexpr bool unpack_to_dest = UNPACKING_TO_DEST;

/*DATA FORMAT CONFIGURATION*/

// Given input and output formats, infer the rest of the format configuration
#if DATA_FORMAT_INFERENCE_MODEL

// If the input is exponentB, we cannot convert it to Float16 without enabling fp32 mode in dest;
// this is considered a format combination outlier, so we enable dest_acc
constexpr bool is_fp32_dest_acc_en =
    dest_acc_en_input || is_format_combination_outlier(static_cast<DataFormat>(UNPACK_A_IN), static_cast<DataFormat>(PACK_OUT), dest_acc_en_input);

// Get Data Formats
inline constexpr std::array<FormatConfig, L1_to_L1_ITERATIONS> formats_array =
    data_formats<static_cast<DataFormat>(UNPACK_A_IN), static_cast<DataFormat>(PACK_OUT), dest_acc_en_input, L1_to_L1_ITERATIONS>();

constexpr auto& formats = formats_array[0];

#else // Not inferring formats — all formats are pre-defined. Set format configuration directly.
constexpr bool is_fp32_dest_acc_en = dest_acc_en_input; // dest_acc doesn't require adjustment; configuration is hard-coded
constexpr FormatConfig formats     = FormatConfig(UNPACK_A_IN, UNPACK_A_OUT, MATH, PACK_IN, PACK_OUT);
#endif
