// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include <cstddef>
#include <ostream>
#include <utility>

#include "shape2d.hpp"

namespace tt::tt_metal {

Shape2D::Shape2D(size_t height, size_t width) : height_(height), width_(width) {}

Shape2D::Shape2D(const std::pair<size_t, size_t>& size) : Shape2D(size.first, size.second) {}

Shape2D::Shape2D(const std::array<size_t, 2>& size) : Shape2D(size[0], size[1]) {}

Shape2D::Shape2D(const std::array<uint32_t, 2>& size) : Shape2D(size[0], size[1]) {}

Shape2D Shape2D::operator*(size_t scalar) const { return Shape2D(height_ * scalar, width_ * scalar); }

Shape2D::operator std::pair<size_t, size_t>() const { return {height_, width_}; }

Shape2D::operator std::array<size_t, 2>() const { return {height_, width_}; }

Shape2D::operator std::array<uint32_t, 2>() const {
    return {static_cast<uint32_t>(height_), static_cast<uint32_t>(width_)};
}

size_t Shape2D::height() const { return height_; }

size_t Shape2D::width() const { return width_; }

bool Shape2D::operator==(const Shape2D& rhs) const { return height_ == rhs.height_ && width_ == rhs.width_; }

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Shape2D& size) {
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

}  // namespace tt::tt_metal
