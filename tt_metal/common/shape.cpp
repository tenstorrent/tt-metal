// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape.hpp"

#include <boost/container/vector.hpp>
#include <boost/move/utility_core.hpp>
#include <tt-metalium/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include <functional>
#include <numeric>
#include <ostream>
#include <utility>

namespace tt::tt_metal {

bool Shape::operator==(const Shape& other) const = default;

bool Shape::operator==(const tt::stl::SmallVector<uint32_t>& other) const { return this->value_ == other; }

size_t Shape::rank() const { return this->size(); }

uint64_t Shape::volume() const { return std::accumulate(cbegin(), cend(), uint64_t{1}, std::multiplies<uint64_t>()); }

std::array<uint32_t, 4> Shape::to_array_4D() const {
    TT_FATAL(rank() == 4, "to_array_4D is only valid for 4D shapes! Called for {}.", *this);
    std::array<uint32_t, 4> ret_array;
    for (int i = 0; i < rank(); i++) {
        ret_array[i] = this->operator[](i);
    }
    return ret_array;
}

Shape Shape::to_rank(size_t new_rank) const {
    tt::stl::SmallVector<uint32_t> new_shape(new_rank, 1);

    int cur_idx = static_cast<int>(rank()) - 1;
    int new_idx = static_cast<int>(new_rank) - 1;
    for (; cur_idx >= 0 && new_idx >= 0; cur_idx--, new_idx--) {
        new_shape[new_idx] = (*this)[cur_idx];
    }
    for (; cur_idx >= 0; cur_idx--) {
        TT_FATAL((*this)[cur_idx] == 1, "Can't convert shape rank");
    }

    return Shape(std::move(new_shape));
}

uint32_t Shape::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank());
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_FATAL(
        normalized_index >= 0 and normalized_index < rank,
        "Index is out of bounds for the rank, should be between 0 and {} however is {}",
        rank - 1,
        normalized_index);
    return normalized_index;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Shape& shape) {
    os << "Shape([";
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";
    return os;
}

tt::stl::SmallVector<uint32_t> compute_strides(const tt::tt_metal::Shape& shape) {
    if (shape.rank() == 0) {
        return {};
    }

    auto num_elements = shape.volume();
    // If any dim is 0, volume would be 0
    if (num_elements == 0) {
        return tt::stl::SmallVector<uint32_t>(shape.rank(), 0);
    }

    tt::stl::SmallVector<uint32_t> strides;
    for (std::int32_t index = 0; index < shape.rank(); index++) {
        num_elements /= shape[index];
        strides.push_back(num_elements);
    }
    return strides;
}

}  // namespace tt::tt_metal

nlohmann::json tt::stl::json::to_json_t<tt::tt_metal::Shape>::operator()(const tt::tt_metal::Shape& shape) const {
    return tt::stl::json::to_json(shape.view());
}

tt::tt_metal::Shape tt::stl::json::from_json_t<tt::tt_metal::Shape>::operator()(
    const nlohmann::json& json_object) const {
    return tt::tt_metal::Shape(tt::stl::json::from_json<tt::tt_metal::ShapeBase::Container>(json_object));
}
