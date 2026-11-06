// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>
#include <utility>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <fmt/base.h>

namespace tt::tt_metal {

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    BFLOAT4_B = 4,
    UINT8 = 5,
    UINT16 = 6,
    INT32 = 7,
    // WARNING: narrowly supported — Blackhole only, ROW-MAJOR only for now, used exclusively
    // by the DeepSeek V3 prefill combine and dispatch ops. Check op support before opting in.
    FP8_E4M3 = 8,
    INT8 = 9,
    // OCP microscaling (MX) formats: 32 elements share one E8M0 scale. Quasar only, TILE layout only.
    MXFP8_E4M3 = 10,
    MXFP8_E5M2 = 11,
    MXFP6_E2M3 = 12,
    MXFP6_E3M2 = 13,
    MXFP4 = 14,
    MXINT8 = 15,
    MXINT4 = 16,
    MXINT2 = 17,
    // GGUF (llama.cpp) quantized weights, one value per format: TILE layout, each page one GGUF record, see GgufTraits.
    GGUF_Q8_0 = 18,
    GGUF_Q4_0 = 19,
    GGUF_IQ4_NL = 20,
    GGUF_MXFP4 = 21,
    GGUF_NVFP4 = 22,
    GGUF_Q6_K = 23,
    GGUF_Q4_K = 24,
    GGUF_Q5_K = 25,
    GGUF_Q3_K = 26,
    GGUF_Q2_K = 27,
    GGUF_IQ4_XS = 28,
    GGUF_TQ2_0 = 29,
    GGUF_Q2_0 = 30,
    GGUF_Q1_0 = 31,
    GGUF_IQ2_XXS = 32,
    GGUF_IQ1_S = 33,
    GGUF_IQ2_S = 34,
    GGUF_IQ3_S = 35,
    GGUF_IQ3_XXS = 36,
    GGUF_IQ2_XS = 37,
    GGUF_IQ1_M = 38,
    GGUF_Q4_1 = 39,
    GGUF_Q5_0 = 40,
    GGUF_Q5_1 = 41,
    GGUF_TQ1_0 = 42,
    INVALID = 43,
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& data_type);

template <typename T>
consteval DataType convert_to_data_type() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DataType::INT8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DataType::UINT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    } else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, ::bfloat16>) {
        return DataType::BFLOAT16;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported DataType!");
    }
}

bool is_floating_point(DataType dtype);

bool is_block_float(DataType dtype);

// True for the MX formats. Kept separate from is_block_float(): BFP and MX share the "TILE-only,
// packed on the host" property but use different encodings, packers and supported architectures.
bool is_mx(DataType dtype);

// True for the GGUF formats. Their pages hold GGUF records, not values: ops that do not handle GGUF see
// DataFormat::Invalid for them, and the host converts them only with from_gguf / to_gguf.
bool is_gguf(DataType dtype);

// One GGUF format. A page holds the record of 32 columns of N by 32 * group_tiles rows of K: their GGUF bytes, with
// the bits regrouped for the Blackhole decoder (scale fields, then or after the codes parts of page_tiles K tiles
// each).
struct GgufTraits {
    uint32_t format_id;    // the decoder's format number (ckernel::gguf::Format)
    uint32_t block_elems;  // elements per GGUF block
    uint32_t block_bytes;  // bytes per GGUF block
    uint32_t group_tiles;  // K tiles per page
    uint32_t page_tiles;   // K tiles per codes part
    uint32_t code_bytes;   // bytes per codes part
    uint32_t scale_rows;   // 64-byte rows of scale fields per page
    uint32_t fp16_rows;    // the last scale rows: fp16 fields, given fp32 copies in the decoder's L1 page
    uint32_t code_offset;  // bytes of grid levels before the codes in the decoder's L1 page (codebook formats)

    uint32_t page_height() const { return 32 * group_tiles; }
    uint32_t page_bytes() const { return 32 * page_height() * block_bytes / block_elems; }

    // The decoder's L1 page of page_tiles K tiles (ckernel::gguf::page_layout): the codes from code_offset, the scale
    // fields from l1_scale_offset, with their fp32 copies l1_scale_bytes in all.
    uint32_t l1_scale_offset() const { return (code_offset + code_bytes + 255) / 256 * 256; }
    uint32_t l1_scale_bytes() const {
        return fp16_rows ? ((scale_rows + 3) / 4 * 4 + 8 * fp16_rows) * 64 : scale_rows * 64;
    }
    uint32_t l1_page_bytes() const {
        uint32_t bytes = 1024;
        while (bytes < l1_scale_offset() + l1_scale_bytes()) {
            bytes *= 2;
        }
        return bytes;
    }
};

const GgufTraits& gguf_traits(DataType dtype);

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);
tt::tt_metal::DataType dataformat_to_datatype_converter(tt::DataFormat dataformat);

/**
 * Returns the format to declare a CB when moving a tensor of this dtype through the unpacker/packer.
 *
 *This differs from datatype_to_dataformat_converter only for INT8.
 */
tt::DataFormat cb_dataformat_for(DataType datatype);

/**
 * Returns tile size of given data type in bytes.
 *
 * Equivalent to tt::tile_size(datatype_to_dataformat_converter(dtype)).
 */
uint32_t tile_size(DataType dtype);

struct NdShardSpec {
    Shape shard_shape;
    CoreRangeSet grid;
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D;

    NdShardSpec with_shard_shape(Shape new_shard_shape) const {
        return NdShardSpec{std::move(new_shard_shape), grid, orientation, shard_distribution_strategy};
    }

    bool operator==(const NdShardSpec& other) const = default;
    bool operator!=(const NdShardSpec& other) const = default;
};

std::ostream& operator<<(std::ostream& os, const NdShardSpec& spec);

}  // namespace tt::tt_metal

template <>
struct fmt::formatter<tt::tt_metal::DataType> : fmt::formatter<string_view> {
    auto format(tt::tt_metal::DataType dt, format_context& ctx) const -> format_context::iterator;
};
