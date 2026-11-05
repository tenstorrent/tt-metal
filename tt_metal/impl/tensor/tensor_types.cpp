// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <string_view>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal {

namespace {

constexpr int kFirstGguf = static_cast<int>(DataType::GGUF_Q8_0);

struct GgufFormat {
    std::string_view name;
    GgufTraits traits;
};

// In DataType order. The numbers match the decoder's page_layout (ckernel_gguf_page.h).
// clang-format off
constexpr std::array<GgufFormat, 25> kGgufFormats{{
    // name                   id  blk  bytes G  page code  scale
    {"DataType::GGUF_Q8_0",    {1, 32, 34, 1, 1, 1024, 1}},
    {"DataType::GGUF_Q4_0",    {2, 32, 18, 1, 1, 512, 1}},
    {"DataType::GGUF_IQ4_NL",  {3, 32, 18, 1, 1, 512, 1}},
    {"DataType::GGUF_MXFP4",   {4, 32, 17, 2, 1, 512, 1}},
    {"DataType::GGUF_NVFP4",   {5, 64, 36, 1, 1, 512, 1}},
    {"DataType::GGUF_Q6_K",    {6, 256, 210, 8, 1, 768, 9}},
    {"DataType::GGUF_Q4_K",    {7, 256, 144, 8, 1, 512, 8}},
    {"DataType::GGUF_Q5_K",    {8, 256, 176, 8, 2, 1280, 8}},
    {"DataType::GGUF_Q3_K",    {9, 256, 110, 8, 2, 768, 7}},
    {"DataType::GGUF_Q2_K",    {10, 256, 84, 8, 1, 256, 10}},
    {"DataType::GGUF_IQ4_XS",  {11, 256, 136, 8, 1, 512, 4}},
    {"DataType::GGUF_TQ2_0",   {12, 256, 66, 8, 1, 256, 1}},
    {"DataType::GGUF_Q2_0",    {13, 64, 18, 2, 1, 256, 1}},
    {"DataType::GGUF_Q1_0",    {14, 128, 18, 4, 2, 256, 1}},
    {"DataType::GGUF_IQ2_XXS", {15, 256, 66, 8, 1, 256, 1}},
    {"DataType::GGUF_IQ1_S",   {16, 256, 50, 8, 1, 192, 1}},
    {"DataType::GGUF_IQ2_S",   {17, 256, 82, 8, 1, 320, 1}},
    {"DataType::GGUF_IQ3_S",   {18, 256, 110, 8, 1, 432, 1}},
    {"DataType::GGUF_IQ3_XXS", {19, 256, 98, 8, 1, 384, 1}},
    {"DataType::GGUF_IQ2_XS",  {20, 256, 74, 8, 1, 256, 5}},
    {"DataType::GGUF_IQ1_M",   {21, 256, 56, 8, 1, 192, 4}},
    {"DataType::GGUF_Q4_1",    {22, 32, 20, 1, 1, 512, 2}},
    {"DataType::GGUF_Q5_0",    {23, 32, 22, 2, 2, 1280, 2}},
    {"DataType::GGUF_Q5_1",    {24, 32, 24, 2, 2, 1280, 4}},
    {"DataType::GGUF_TQ1_0",   {25, 256, 54, 8, 8, 1536, 3}},
}};
// clang-format on
static_assert(kFirstGguf + static_cast<int>(kGgufFormats.size()) == static_cast<int>(DataType::INVALID));

const GgufFormat& gguf_format(DataType dtype) {
    TT_FATAL(is_gguf(dtype), "{} is not a GGUF dtype", dtype);
    return kGgufFormats[static_cast<int>(dtype) - kFirstGguf];
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& data_type) {
    switch (data_type) {
        case DataType::BFLOAT16: return os << "DataType::BFLOAT16";
        case DataType::FLOAT32: return os << "DataType::FLOAT32";
        case DataType::UINT32: return os << "DataType::UINT32";
        case DataType::BFLOAT8_B: return os << "DataType::BFLOAT8_B";
        case DataType::BFLOAT4_B: return os << "DataType::BFLOAT4_B";
        case DataType::UINT8: return os << "DataType::UINT8";
        case DataType::UINT16: return os << "DataType::UINT16";
        case DataType::INT32: return os << "DataType::INT32";
        case DataType::FP8_E4M3: return os << "DataType::FP8_E4M3";
        case DataType::INT8: return os << "DataType::INT8";
        case DataType::MXFP8_E4M3: return os << "DataType::MXFP8_E4M3";
        case DataType::MXFP8_E5M2: return os << "DataType::MXFP8_E5M2";
        case DataType::MXFP6_E2M3: return os << "DataType::MXFP6_E2M3";
        case DataType::MXFP6_E3M2: return os << "DataType::MXFP6_E3M2";
        case DataType::MXFP4: return os << "DataType::MXFP4";
        case DataType::MXINT8: return os << "DataType::MXINT8";
        case DataType::MXINT4: return os << "DataType::MXINT4";
        case DataType::MXINT2: return os << "DataType::MXINT2";
        case DataType::INVALID: return os << "Invalid";
        default: return os << (is_gguf(data_type) ? gguf_format(data_type).name : "Invalid");
    }
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::NdShardSpec& spec) {
    os << "{";
    os << "\"shard_shape\":[";

    // Format shard_shape as array
    const auto& shape = spec.shard_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "],";

    os << "\"grid\":[";

    const auto& ranges = spec.grid.ranges();
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto& range = ranges[i];
        os << R"({"start":{"x":)" << range.start_coord.x << ",\"y\":" << range.start_coord.y << "},";
        os << R"("end":{"x":)" << range.end_coord.x << ",\"y\":" << range.end_coord.y << "}}";
        if (i < ranges.size() - 1) {
            os << ", ";
        }
    }
    os << "],";

    os << R"("orientation":")";
    switch (spec.orientation) {
        case ShardOrientation::ROW_MAJOR: os << "ShardOrientation::ROW_MAJOR"; break;
        case ShardOrientation::COL_MAJOR: os << "ShardOrientation::COL_MAJOR"; break;
    }
    os << "\",";

    os << R"("shard_distribution_strategy":")";
    switch (spec.shard_distribution_strategy) {
        case ShardDistributionStrategy::ROUND_ROBIN_1D: os << "ShardDistributionStrategy::ROUND_ROBIN_1D"; break;
        case ShardDistributionStrategy::GRID_2D: os << "ShardDistributionStrategy::GRID_2D"; break;
        case ShardDistributionStrategy::CONTIGUOUS_1D: os << "ShardDistributionStrategy::CONTIGUOUS_1D"; break;
    }
    os << "\"";

    os << "}";
    return os;
}

bool is_floating_point(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
        case DataType::FP8_E4M3:
        // MXINT elements are integers, but every MX value carries a block exponent and the hardware
        // computes on them as Float16_b, so all MX formats count as floating point.
        case DataType::MXFP8_E4M3:
        case DataType::MXFP8_E5M2:
        case DataType::MXFP6_E2M3:
        case DataType::MXFP6_E3M2:
        case DataType::MXFP4:
        case DataType::MXINT8:
        case DataType::MXINT4:
        case DataType::MXINT2: return true;
        default: return false;
    }
}

bool is_block_float(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return true;
        default: return false;
    }
}

bool is_gguf(DataType dtype) { return static_cast<int>(dtype) >= kFirstGguf && dtype < DataType::INVALID; }

const GgufTraits& gguf_traits(DataType dtype) { return gguf_format(dtype).traits; }

bool is_mx(DataType dtype) {
    switch (dtype) {
        case DataType::MXFP8_E4M3:
        case DataType::MXFP8_E5M2:
        case DataType::MXFP6_E2M3:
        case DataType::MXFP6_E3M2:
        case DataType::MXFP4:
        case DataType::MXINT8:
        case DataType::MXINT4:
        case DataType::MXINT2: return true;
        default: return false;
    }
}

tt::DataFormat datatype_to_dataformat_converter(tt::tt_metal::DataType datatype) {
    switch (datatype) {
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
        case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
        case tt::tt_metal::DataType::INT32: return tt::DataFormat::Int32;
        case tt::tt_metal::DataType::INT8: return tt::DataFormat::Int8;
        case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
        case tt::tt_metal::DataType::UINT16: return tt::DataFormat::UInt16;
        case tt::tt_metal::DataType::UINT8: return tt::DataFormat::UInt8;
        case tt::tt_metal::DataType::FP8_E4M3: return tt::DataFormat::Fp8_e4m3;
        // tt-metal names the MX variants R ("range", more exponent bits) and P ("precision").
        case tt::tt_metal::DataType::MXFP8_E4M3: return tt::DataFormat::MxFp8P;
        case tt::tt_metal::DataType::MXFP8_E5M2: return tt::DataFormat::MxFp8R;
        case tt::tt_metal::DataType::MXFP6_E2M3: return tt::DataFormat::MxFp6P;
        case tt::tt_metal::DataType::MXFP6_E3M2: return tt::DataFormat::MxFp6R;
        case tt::tt_metal::DataType::MXFP4: return tt::DataFormat::MxFp4;
        case tt::tt_metal::DataType::MXINT8: return tt::DataFormat::MxInt8;
        case tt::tt_metal::DataType::MXINT4: return tt::DataFormat::MxInt4;
        case tt::tt_metal::DataType::MXINT2: return tt::DataFormat::MxInt2;
        default:
            // GGUF pages are records, not values that any unpacker can read.
            if (is_gguf(datatype)) {
                return tt::DataFormat::Invalid;
            }
            TT_THROW("Unsupported DataType");
            return tt::DataFormat::Float16_b;  // for clang-tidy
    }
}

tt::DataFormat cb_dataformat_for(tt::tt_metal::DataType datatype) {
    return datatype == tt::tt_metal::DataType::INT8 ? tt::DataFormat::UInt8
                                                    : datatype_to_dataformat_converter(datatype);
}

tt::tt_metal::DataType dataformat_to_datatype_converter(tt::DataFormat dataformat) {
    switch (dataformat) {
        case tt::DataFormat::Float16_b: return tt::tt_metal::DataType::BFLOAT16;
        case tt::DataFormat::Bfp8_b: return tt::tt_metal::DataType::BFLOAT8_B;
        case tt::DataFormat::Bfp4_b: return tt::tt_metal::DataType::BFLOAT4_B;
        case tt::DataFormat::Float32: return tt::tt_metal::DataType::FLOAT32;
        case tt::DataFormat::Int32: return tt::tt_metal::DataType::INT32;
        case tt::DataFormat::Int8: return tt::tt_metal::DataType::INT8;
        case tt::DataFormat::UInt32: return tt::tt_metal::DataType::UINT32;
        case tt::DataFormat::UInt16: return tt::tt_metal::DataType::UINT16;
        case tt::DataFormat::UInt8: return tt::tt_metal::DataType::UINT8;
        case tt::DataFormat::Fp8_e4m3: return tt::tt_metal::DataType::FP8_E4M3;
        case tt::DataFormat::MxFp8P: return tt::tt_metal::DataType::MXFP8_E4M3;
        case tt::DataFormat::MxFp8R: return tt::tt_metal::DataType::MXFP8_E5M2;
        case tt::DataFormat::MxFp6P: return tt::tt_metal::DataType::MXFP6_E2M3;
        case tt::DataFormat::MxFp6R: return tt::tt_metal::DataType::MXFP6_E3M2;
        case tt::DataFormat::MxFp4: return tt::tt_metal::DataType::MXFP4;
        case tt::DataFormat::MxInt8: return tt::tt_metal::DataType::MXINT8;
        case tt::DataFormat::MxInt4: return tt::tt_metal::DataType::MXINT4;
        case tt::DataFormat::MxInt2: return tt::tt_metal::DataType::MXINT2;
        default: TT_THROW("Unsupported DataFormat"); return tt::tt_metal::DataType::BFLOAT16;  // for clang-tidy
    }
}

uint32_t tile_size(DataType dtype) {
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
    return tt::tile_size(output_data_format);
}

}  // namespace tt::tt_metal

auto fmt::formatter<tt::tt_metal::DataType>::format(tt::tt_metal::DataType dt, format_context& ctx) const
    -> format_context::iterator {
    string_view name;
    switch (dt) {
        case tt::tt_metal::DataType::BFLOAT16: name = "DataType::BFLOAT16"; break;
        case tt::tt_metal::DataType::FLOAT32: name = "DataType::FLOAT32"; break;
        case tt::tt_metal::DataType::UINT32: name = "DataType::UINT32"; break;
        case tt::tt_metal::DataType::BFLOAT8_B: name = "DataType::BFLOAT8_B"; break;
        case tt::tt_metal::DataType::BFLOAT4_B: name = "DataType::BFLOAT4_B"; break;
        case tt::tt_metal::DataType::UINT8: name = "DataType::UINT8"; break;
        case tt::tt_metal::DataType::UINT16: name = "DataType::UINT16"; break;
        case tt::tt_metal::DataType::INT32: name = "DataType::INT32"; break;
        case tt::tt_metal::DataType::FP8_E4M3: name = "DataType::FP8_E4M3"; break;
        case tt::tt_metal::DataType::INT8: name = "DataType::INT8"; break;
        case tt::tt_metal::DataType::MXFP8_E4M3: name = "DataType::MXFP8_E4M3"; break;
        case tt::tt_metal::DataType::MXFP8_E5M2: name = "DataType::MXFP8_E5M2"; break;
        case tt::tt_metal::DataType::MXFP6_E2M3: name = "DataType::MXFP6_E2M3"; break;
        case tt::tt_metal::DataType::MXFP6_E3M2: name = "DataType::MXFP6_E3M2"; break;
        case tt::tt_metal::DataType::MXFP4: name = "DataType::MXFP4"; break;
        case tt::tt_metal::DataType::MXINT8: name = "DataType::MXINT8"; break;
        case tt::tt_metal::DataType::MXINT4: name = "DataType::MXINT4"; break;
        case tt::tt_metal::DataType::MXINT2: name = "DataType::MXINT2"; break;
        case tt::tt_metal::DataType::INVALID: name = "Invalid"; break;
        default: name = tt::tt_metal::is_gguf(dt) ? tt::tt_metal::gguf_format(dt).name : "Invalid"; break;
    }
    return formatter<string_view>::format(name, ctx);
}
