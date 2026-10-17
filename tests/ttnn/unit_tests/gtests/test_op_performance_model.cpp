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
    // At 1 MiB, scaling the GB/s peak by 2^30 instead of 1e9 is off by about 7%, far more than the
    // integer truncation of the model.
    const auto tensor = ttnn::zeros(
        ttnn::Shape({1, 1, 512, 1024}), DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_, ttnn::DRAM_MEMORY_CONFIG);
    const tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> model({tensor}, tensor, 1);

    const double peak_dram_gb_per_s = device_->arch() == tt::ARCH::BLACKHOLE ? 512.0 : 258.0;
    const double size_bytes = static_cast<double>(tensor.physical_volume() * tensor.element_size());
    EXPECT_EQ(model.get_bandwidth_ns(), static_cast<int>(size_bytes / peak_dram_gb_per_s));
}

}  // namespace ttnn::test
