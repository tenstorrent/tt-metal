// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <ostream>

namespace tt::tt_metal::distributed {

struct MeshShape {
    size_t num_rows = 0;
    size_t num_cols = 0;
};

struct MeshOffset {
    size_t row = 0;
    size_t col = 0;
};

struct Coordinate {
    size_t row = 0;
    size_t col = 0;
    auto operator<=>(const Coordinate&) const = default;

    template <size_t I>
    decltype(auto) get() const {
        if constexpr (I == 0) {
            return row;
        } else if constexpr (I == 1) {
            return col;
        } else {
            static_assert(I < 2, "Index out of bounds for Coordinate");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Coordinate& coord) {
        return os << "Coord(" << coord.row << ", " << coord.col << ")";
    }
};

}  // namespace tt::tt_metal::distributed
