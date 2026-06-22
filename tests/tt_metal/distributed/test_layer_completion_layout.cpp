// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <type_traits>

#include <internal/service/layer_completion_message.hpp>
#include "layer_completion_ring_layout.hpp"

namespace tt::tt_metal::distributed::test {

TEST(LayerCompletionLayout, MessageIsTrivialAndStable) {
    static_assert(std::is_trivially_copyable_v<LayerCompletionMessage>);
    EXPECT_EQ(sizeof(LayerCompletionMessage), 24u);
    EXPECT_EQ(alignof(LayerCompletionMessage), 8u);
}

TEST(LayerCompletionLayout, CapacityIsPowerOfTwo) {
    EXPECT_NE(kLayerCompletionRingCapacity, 0u);
    EXPECT_EQ(kLayerCompletionRingCapacity & (kLayerCompletionRingCapacity - 1), 0u);
    EXPECT_EQ(kLayerCompletionRingMask, kLayerCompletionRingCapacity - 1);
}

TEST(LayerCompletionLayout, RingBytesCoversHeaderAndAllCells) {
    EXPECT_GE(layer_completion_cells_offset(), sizeof(LayerCompletionRingHeader));
    EXPECT_EQ(
        kLayerCompletionRingBytes,
        layer_completion_cells_offset() +
            static_cast<std::size_t>(kLayerCompletionRingCapacity) * sizeof(LayerCompletionCell));
}

}  // namespace tt::tt_metal::distributed::test
