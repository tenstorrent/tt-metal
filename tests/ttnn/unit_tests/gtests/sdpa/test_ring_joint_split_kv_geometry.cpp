// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the split-KV geometry the ring_joint SDPA validates (ring_mla_geometry.hpp).

#include <cstdint>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_geometry.hpp"

namespace {

using namespace ttnn::operations::transformer::sdpa::ring_joint;

TEST(RingMLAGeometry, AcceptsWholeRegionsAndRejectsTheRest) {
    // Galaxy SP8 x TP4 production: Q slab 20 tiles, four stripes of 5-tile regions.
    EXPECT_TRUE((RingMLAGeometry{8, 32, 20, 55}.valid()));
    // LB SP2 x TP4.
    EXPECT_TRUE((RingMLAGeometry{2, 8, 8, 16}.valid()));
    // Legacy equal-shard layout: one stripe.
    EXPECT_TRUE((RingMLAGeometry{32, 32, 2, 4}.valid()));

    EXPECT_FALSE((RingMLAGeometry{0, 32, 20, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 0, 20, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 64, 20, 55}.valid()));  // more than 32 sources
    EXPECT_FALSE((RingMLAGeometry{3, 32, 20, 55}.valid()));  // sources not a multiple of Q shards
    EXPECT_FALSE((RingMLAGeometry{8, 32, 0, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 22, 55}.valid()));  // Q slab not a whole number of stripes
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 0}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 56}.valid()));  // capacity not whole regions
    // Whole regions, but 32 sources of 134217730 tiles overflow uint32_t.
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 134217730}.valid()));
}

}  // namespace
