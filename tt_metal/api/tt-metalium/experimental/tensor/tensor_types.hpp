// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file is a copy of TTNN's ttnn/api/ttnn/tensor/types.hpp
// at commit 9f3856801448f589170defe41b23c8b9b43e33a2, with modifications to
// use experimental tensor types.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include <iostream>

#include <tt-metalium/experimental/tensor/spec/shape/shape.hpp>  // For Shape
#include <tt-metalium/experimental/tensor/spec/memory_config/sharding_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/concepts.hpp>

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
    INVALID = 8,
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& data_type);

template <typename T>
consteval DataType convert_to_data_type() {
    if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UINT8;
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
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType!");
    }
}

constexpr std::size_t data_type_size(DataType type) {
    switch (type) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: return sizeof(std::byte);
        default: TT_THROW("Unsupported data type");
    }
}

bool is_floating_point(DataType dtype);

bool is_block_float(DataType dtype);

// Specifies Tensor storage type.
enum class StorageType {
    HOST = 0,
    DEVICE = 1,
};

tt::DataFormat datatype_to_dataformat_converter(DataType datatype);
tt::tt_metal::DataType dataformat_to_datatype_converter(tt::DataFormat dataformat);

static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

using Array1D = std::array<uint32_t, 1>;
using Array2D = std::array<uint32_t, 2>;
using Array3D = std::array<uint32_t, 3>;
using Array4D = std::array<uint32_t, 4>;
using Array5D = std::array<uint32_t, 5>;
using Array6D = std::array<uint32_t, 6>;
using Array7D = std::array<uint32_t, 7>;
using Array8D = std::array<uint32_t, 8>;

using PadValue = std::variant<uint32_t, float>;

enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

}  // namespace tt::tt_metal
