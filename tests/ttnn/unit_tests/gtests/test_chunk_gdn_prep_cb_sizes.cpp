// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased.hpp"

namespace {

TEST(ChunkGdnPrepCbSizes, S3RightSizingPreservesPrecedingLayout) {
    constexpr auto capacities = ttnn::prim::detail::chunk_gdn_prep_cb_tile_capacities(1, 4, 4);
    constexpr std::array<uint32_t, 31> legacy_prefix = {
        4, 4, 4, 1, 1, 1, 1, 1, 32, 1, 1, 1, 1, 1, 4, 4, 8, 4, 4, 4, 1, 32, 4, 4, 4, 16, 16, 16, 16, 16, 16};

    for (std::size_t i = 0; i < legacy_prefix.size(); ++i) {
        EXPECT_EQ(capacities[i], legacy_prefix[i]) << "CB index " << i << " moved or changed capacity";
    }
    EXPECT_EQ(capacities[31], 1u);
}

TEST(ChunkGdnPrepCbSizes, K128V128ClearsObservedCallerAllocation) {
    constexpr uint32_t cb_base = 111616;
    constexpr uint32_t caller_allocation = 994816;
    constexpr uint32_t cb_bytes = ttnn::prim::detail::chunk_gdn_prep_cb_bytes(1, 4, 4);

    static_assert(cb_bytes == 876544);
    static_assert(cb_base + cb_bytes == 988160);
    static_assert(cb_base + cb_bytes < caller_allocation);
    EXPECT_EQ(caller_allocation - (cb_base + cb_bytes), 6656u);
}

}  // namespace
