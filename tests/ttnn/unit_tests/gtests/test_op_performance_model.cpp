// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/shape.hpp>
#include "api/ttnn/operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

using OpPerformanceModelDramTest = TTNNFixtureWithDevice;

TEST_F(OpPerformanceModelDramTest, BandwidthUsesDecimalGigabytesPerSecond) {
    // bfloat16 [1, 1, 32, 64] is 4096 bytes. Scaling the GB/s peak by 2^30 instead of 1e9 truncates to
    // 14 ns on Wormhole (258 GB/s) and 7 ns on Blackhole (512 GB/s) instead of 15 and 8.
    const auto tensor = ttnn::zeros(
        ttnn::Shape({1, 1, 32, 64}), DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_, ttnn::DRAM_MEMORY_CONFIG);
    const tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> model({tensor}, tensor, 1);

    const int expected_ns = device_->arch() == tt::ARCH::BLACKHOLE ? 8 : 15;
    EXPECT_EQ(model.get_bandwidth_ns(), expected_ns);
}

}  // namespace ttnn::test
