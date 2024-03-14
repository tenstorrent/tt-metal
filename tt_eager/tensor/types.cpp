// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/types.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <boost/core/demangle.hpp>

namespace tt {

namespace tt_metal {


tt::DataFormat datatype_to_dataformat_converter(tt::tt_metal::DataType datatype) {
    switch (datatype) {
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
        case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
        case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
        case tt::tt_metal::DataType::UINT16: return tt::DataFormat::UInt16;
        default:
            TT_ASSERT(false, "Unsupported DataType");
            return tt::DataFormat::Float16_b;
    }
}

Padding::Padding(const std::size_t rank) : rank_{rank}, pad_dimensions_{}, pad_value_{} {}

Padding::Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value) :
    rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
    std::copy(std::begin(pad_dimensions), std::end(pad_dimensions), std::begin(this->pad_dimensions_));
}

Padding::Padding(const std::vector<PadDimension>& pad_dimensions, PadValue pad_value) :
    rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
    std::copy(std::begin(pad_dimensions), std::end(pad_dimensions), std::begin(this->pad_dimensions_));
}

Padding::PadDimension& Padding::operator[](const std::int64_t index) {
    return this->pad_dimensions_[index];
}

const Padding::PadDimension& Padding::operator[](const std::int64_t index) const  {
    return this->pad_dimensions_[index];
}

Padding::PadValue Padding::pad_value() const { return this->pad_value_; }

bool operator==(const Padding& padding_a, const Padding& padding_b) {
    if (padding_a.rank_ != padding_b.rank_) {
        return false;
    }
    for (auto index = 0; index < padding_a.rank_; index++) {
        if (padding_a[index].front != padding_b[index].front) {
            return false;
        }

        if (padding_a[index].back != padding_b[index].back) {
            return false;
        }
    }
    return padding_a.pad_value_ == padding_b.pad_value_;
}

bool operator!=(const Padding& padding_a, const Padding& padding_b) { return not(padding_a == padding_b); }

Shape::Shape(const std::initializer_list<uint32_t> dimensions) :
    rank_(dimensions.size()), dimensions_{}, padding_(dimensions.size()) {
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}
Shape::Shape(const std::vector<uint32_t>& dimensions) :
    rank_(dimensions.size()), dimensions_{}, padding_(dimensions.size()) {
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}

Shape::Shape(const std::initializer_list<uint32_t> dimensions, const Padding& padding) :
    rank_(dimensions.size()), dimensions_{}, padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}
Shape::Shape(const std::vector<uint32_t>& dimensions, const Padding& padding) :
    rank_(dimensions.size()), dimensions_{}, padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
    std::copy(std::begin(dimensions), std::end(dimensions), std::begin(this->dimensions_));
}

Shape::Shape(const Shape& other, const Padding& padding) :
    dimensions_(other.dimensions_), rank_(other.rank_), padding_(padding) {
    TT_ASSERT(this->padding_.rank_ == this->rank_);
}

std::size_t Shape::rank() const { return this->rank_; }

uint32_t& Shape::operator[](const std::int64_t index) {
    auto normalized_index = this->get_normalized_index(index);
    return this->dimensions_[normalized_index];
}
const uint32_t& Shape::operator[](const std::int64_t index) const {
    auto normalized_index = this->get_normalized_index(index);
    return this->dimensions_[normalized_index];
}

const uint32_t* Shape::begin() const { return this->dimensions_.data(); }
const uint32_t* Shape::end() const { return this->dimensions_.data() + this->rank_; }

const Padding& Shape::padding() const {
    return this->padding_;
}

const Shape Shape::without_padding() const {
    auto padding = this->padding_;
    std::vector<std::uint32_t> shape_without_padding;
    for (auto index = 0; index < this->rank(); index++) {
        const auto dimension = this->operator[](index);
        auto&& [front_pad, back_pad] = padding.pad_dimensions_[index];
        const auto new_dimension = dimension - (front_pad + back_pad);
        shape_without_padding.push_back(new_dimension);
    }
    return Shape(shape_without_padding);
}

const uint32_t Shape::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank_);
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_ASSERT(
        normalized_index >= 0 and normalized_index < rank,
        fmt::format(
            "Index is out of bounds for the rank, should be between 0 and {} however is {}",
            rank - 1,
            normalized_index));
    return normalized_index;
}

bool operator==(const Shape& shape_a, const Shape& shape_b) {
    if (shape_a.rank() != shape_b.rank()) {
        return false;
    }
    for (auto index = 0; index < shape_a.rank(); index++) {
        if (shape_a[index] != shape_b[index]) {
            return false;
        }
    }
    // TODO:(arakhmati): should the padding be ignored?
    return true;  // Ignore the padding when comparing shapes
}

bool operator!=(const Shape& shape_a, const Shape& shape_b) { return not(shape_a == shape_b); }

bool MemoryConfig::is_sharded() const {
    switch (this->memory_layout) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        case TensorMemoryLayout::WIDTH_SHARDED:
        case TensorMemoryLayout::BLOCK_SHARDED: return true;
        default: return false;
    }
}

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b) {
    return config_a.buffer_type == config_b.buffer_type && config_a.memory_layout == config_b.memory_layout && config_a.shard_spec == config_b.shard_spec;
}

bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b) { return not(config_a == config_b); }

}  // namespace tt_metal

}  // namespace tt
