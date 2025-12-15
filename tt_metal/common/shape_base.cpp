// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <vector>

#include "shape_base.hpp"
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

namespace detail {
[[gnu::noinline]] void normalized_index_out_of_range(int32_t index, int32_t full_size, int32_t orig_size) {
    TT_THROW("ShapeBase[] index out of range. {} not in [{}, {})", index, -full_size, orig_size);
}
}  // namespace detail

void ShapeBase::init() {
    original_size_ = value_.size();
    const size_t min_internal_size = 4;

    if (original_size_ < min_internal_size) {
        Container ones(min_internal_size - original_size_, 1);
        value_.insert(value_.begin(), ones.begin(), ones.end());
    }
}

bool ShapeBase::empty() const { return original_size_ == 0; }

size_t ShapeBase::size() const { return original_size_; }

tt::stl::Span<const uint32_t> ShapeBase::view() const { return tt::stl::Span<const uint32_t>{cbegin(), cend()}; }

bool ShapeBase::operator==(const ShapeBase& other) const = default;

bool ShapeBase::operator==(const Container& other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

bool ShapeBase::operator==(const std::vector<uint32_t>& other) const {
    auto original_view = view();
    return std::equal(original_view.begin(), original_view.end(), other.begin(), other.end());
}

ShapeBase::Container::const_iterator ShapeBase::cbegin() const {
    return this->value_.cbegin() + (value_.size() - original_size_);
}

ShapeBase::Container::const_iterator ShapeBase::cend() const { return this->value_.cend(); }

}  // namespace tt::tt_metal
