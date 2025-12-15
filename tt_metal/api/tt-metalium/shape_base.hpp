// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt_stl/small_vector.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// Inline implementation of operator[] index normalization to enable compiler optimization.
// This allows the compiler to optimize hot loops that repeatedly index shapes, eliminating
// function call overhead.
namespace detail {
[[noreturn]] void normalized_index_out_of_range(int32_t index, int32_t full_size, int32_t orig_size);

inline int32_t normalized_index(int32_t index, size_t original_size, size_t container_size) {
    const int32_t orig_size = static_cast<int32_t>(original_size);
    const int32_t full_size = static_cast<int32_t>(container_size);

    if (index < -full_size || index >= orig_size) {
        normalized_index_out_of_range(index, full_size, orig_size);
    }

    const int32_t is_nonneg = index >= 0;
    const int32_t adjust = full_size - is_nonneg * orig_size;
    return index + adjust;
}
}  // namespace detail

// Container wrapper that allows negative indexing
class ShapeBase {
public:
    using Container = tt::stl::SmallVector<uint32_t>;

    ShapeBase() { init(); };
    explicit ShapeBase(const Container& shape) : value_(shape) { init(); }
    explicit ShapeBase(Container&& shape) : value_(std::move(shape)) { init(); }
    explicit ShapeBase(std::initializer_list<uint32_t> ilist) : value_(ilist) { init(); }
    template <std::size_t N>
    explicit ShapeBase(const std::array<uint32_t, N>& arr) : value_(arr.begin(), arr.end()) {
        init();
    }
    explicit ShapeBase(tt::stl::Span<const uint32_t> span) : value_(span.begin(), span.end()) { init(); }

    template <std::size_t N>
    bool operator==(const std::array<uint32_t, N>& other) const {
        bool same_size = value_.size() == N;
        return same_size && std::equal(value_.begin(), value_.end(), other.begin());
    }

    bool operator==(const ShapeBase& other) const;
    bool operator==(const Container& other) const;
    bool operator==(const std::vector<uint32_t>& other) const;

    uint32_t operator[](int32_t index) const {
        auto norm_index = detail::normalized_index(index, original_size_, value_.size());
        return value_[norm_index];
    }

    uint32_t& operator[](int32_t index) {
        auto norm_index = detail::normalized_index(index, original_size_, value_.size());
        return value_[norm_index];
    }

    Container::const_iterator cbegin() const;
    Container::const_iterator cend() const;

    tt::stl::Span<const uint32_t> view() const;

    bool empty() const;

protected:
    void init();
    size_t size() const;

    Container value_;

private:
    size_t original_size_ = 0;
};

}  // namespace tt::tt_metal
