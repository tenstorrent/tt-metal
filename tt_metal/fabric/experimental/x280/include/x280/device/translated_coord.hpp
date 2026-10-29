// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "x280_noc_coord.h"

#include <cstdint>

namespace x280 {

class X280Runtime;

namespace testing {
struct TranslatedCoordFactory;
}

/**
 * A NoC coordinate in the translated (virtual) space firmware configures
 * (BlackholeA0/NoC/Coordinates.md, "Coordinate Translation"), resolved for the
 * NoC its consumer uses. Only X280Runtime's resolvers construct one, so every
 * value that reaches a device argument came through them: a logical or NOC0
 * value cannot. Both axes pass x280_noc_coord_valid.
 *
 * x280::testing::TranslatedCoordFactory, defined only in test code, is the one
 * other constructor; host tests use it in place of a device.
 */
class TranslatedCoord {
public:
    uint32_t x() const { return coord_.x; }
    uint32_t y() const { return coord_.y; }
    /** The value as device arguments carry it. */
    x280_noc_coord_t noc() const { return coord_; }

    bool operator==(const TranslatedCoord& o) const { return coord_.x == o.coord_.x && coord_.y == o.coord_.y; }
    bool operator!=(const TranslatedCoord& o) const { return !(*this == o); }

private:
    friend class X280Runtime;
    friend struct testing::TranslatedCoordFactory;

    constexpr TranslatedCoord(uint32_t x, uint32_t y) : coord_{x, y} {}

    x280_noc_coord_t coord_;
};

}  // namespace x280
