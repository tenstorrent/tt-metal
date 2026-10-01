// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/tensor/tensor_types.hpp>

namespace tt::tt_metal {

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
        case DataType::INVALID:
        default: return os << "Invalid";
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
        default: TT_THROW("Unsupported DataType"); return tt::DataFormat::Float16_b;  // for clang-tidy
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
        case tt::tt_metal::DataType::INVALID:
        default: name = "Invalid"; break;
    }
    return formatter<string_view>::format(name, ctx);
}
