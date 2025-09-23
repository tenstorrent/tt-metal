// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

// Include auto-generated build configuration
#include "build.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "data_format_inference.h"
#include "perf.h"
#include "tensix_types.h"

inline uint32_t L1_ADDRESS(uint32_t buffer_address)
{
    return (buffer_address / 16) - 1;
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
constexpr FormatConfig formats     = FormatConfig(UNPACK_A_IN, UNPACK_A_OUT, MATH_FORMAT, PACK_IN, PACK_OUT);
#endif

// Tile count validation - applies to all kernel variants (UNPACK, MATH, PACK)
#if defined(RT_DIM) && defined(CT_DIM)
constexpr uint32_t tile_count          = RT_DIM * CT_DIM;
constexpr uint32_t max_tiles_fp32_dest = 4; // 32-bit dest accumulation limit
constexpr uint32_t max_tiles_fp16_dest = 8; // 16-bit dest accumulation limit

static_assert(tile_count > 0, "Matrix dimensions invalid: RT_DIM and CT_DIM must be positive");

static_assert(tile_count <= max_tiles_fp16_dest, "Tile count exceeds hardware limit: RT_DIM * CT_DIM must be <= 8");

static_assert(
    !is_fp32_dest_acc_en || (tile_count <= max_tiles_fp32_dest),
    "FP32 dest accumulation requires RT_DIM * CT_DIM <= 4 (current configuration exceeds this limit)");
#endif
