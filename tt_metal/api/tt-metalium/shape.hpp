// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <tuple>

#include <tt-metalium/shape_base.hpp>
#include <tt_stl/small_vector.hpp>

namespace tt::tt_metal {

class Shape final : protected ShapeBase {
public:
    using ShapeBase::ShapeBase;
    using ShapeBase::operator[];
    using ShapeBase::cbegin;
    using ShapeBase::cend;
    using ShapeBase::empty;
    using ShapeBase::size;
    using ShapeBase::view;

    template <std::size_t N>
    bool operator==(const std::array<uint32_t, N>& other) const {
        const bool sameSize = value_.size() == N;
        return sameSize && std::equal(value_.begin(), value_.end(), other.begin());
    }

    bool operator==(const Shape& other) const;
    bool operator==(const ShapeBase::Container& other) const;

    [[nodiscard]] size_t rank() const;
    [[nodiscard]] uint64_t volume() const;

    uint32_t get_normalized_index(std::int64_t index) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->value_); }

    std::array<uint32_t, 4> to_array_4D() const;
    Shape to_rank(size_t new_rank) const;

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape);
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Shape& shape);

tt::stl::SmallVector<size_t> compute_strides(const tt::tt_metal::Shape& shape);

}  // namespace tt::tt_metal

template <>
struct ttsl::json::to_json_t<tt::tt_metal::Shape> {
    nlohmann::json operator()(const tt::tt_metal::Shape& shape) const;
};

template <>
struct ttsl::json::from_json_t<tt::tt_metal::Shape> {
    tt::tt_metal::Shape operator()(const nlohmann::json& json_object) const;
};
