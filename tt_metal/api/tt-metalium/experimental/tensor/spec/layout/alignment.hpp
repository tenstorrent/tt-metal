// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <fmt/core.h>

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/shape_base.hpp>

namespace tt::tt_metal {

class Alignment final : protected ShapeBase {
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

    bool operator==(const Alignment& other) const;
    bool operator==(const ttsl::SmallVector<uint32_t>& other) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->value_); }

    friend std::ostream& operator<<(std::ostream& os, const Alignment& alignment);
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Alignment& alignment);

}  // namespace tt::tt_metal

// Out-of-line string conversion (defined in alignment.cpp).
namespace ttsl::fmt_detail {
std::string to_string(const tt::tt_metal::Alignment& alignment);
}  // namespace ttsl::fmt_detail

// Lightweight fmt::formatter – delegates to out-of-line to_string().
template <>
struct fmt::formatter<tt::tt_metal::Alignment> : fmt::formatter<std::string_view> {
    auto format(const tt::tt_metal::Alignment& val, fmt::format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(ttsl::fmt_detail::to_string(val), ctx);
    }
};
