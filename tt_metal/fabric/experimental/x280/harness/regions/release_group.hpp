// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_HARNESS_REGIONS_RELEASE_GROUP_HPP
#define X280_HARNESS_REGIONS_RELEASE_GROUP_HPP

#include "x280/device/translated_coord.hpp"
#include "x280_noc_coord.h"

#include <cstdint>

namespace x280::harness::regions {

/** One multicast the release selector chose: the virtual images of the
 * selected logical rectangle's two corners, and how many participants it
 * releases (the logical Tensix count, excluding a non-loopback sender). */
struct ReleaseGroup {
    TranslatedCoord start;
    TranslatedCoord end;
    uint32_t num_dests = 0;

    x280_noc_rect_t rect() const { return {start.noc(), end.noc()}; }
};

}  // namespace x280::harness::regions

#endif
