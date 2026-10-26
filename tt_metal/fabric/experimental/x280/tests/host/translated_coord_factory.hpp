// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The test-only TranslatedCoord constructor, standing in for the resolver in
// host tests that have no device.

#ifndef X280_TESTS_HOST_TRANSLATED_COORD_FACTORY_HPP
#define X280_TESTS_HOST_TRANSLATED_COORD_FACTORY_HPP

#include "x280/device/translated_coord.hpp"

#include <stdexcept>
#include <string>

namespace x280::testing {

struct TranslatedCoordFactory {
    /** Throws std::invalid_argument for a value the resolver would refuse. */
    static TranslatedCoord make(uint32_t x, uint32_t y) {
        if (!x280_noc_coord_valid({x, y})) {
            throw std::invalid_argument(
                "test coordinate (" + std::to_string(x) + ", " + std::to_string(y) + ") exceeds six bits");
        }
        return TranslatedCoord(x, y);
    }
};

}  // namespace x280::testing

#endif
