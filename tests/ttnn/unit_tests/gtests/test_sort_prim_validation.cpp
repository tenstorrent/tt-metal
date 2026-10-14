// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <exception>
#include <optional>
#include <vector>

#include "ttnn/device.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/sort/device/sort_device_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::prim::test {

class SortPrimValidationFixture : public TTNNFixtureWithSuiteDevice<SortPrimValidationFixture> {};

// The bitonic sort engines require a power-of-two sort width: they have no
// j < Wt partner guard and truncate log2(Wt), so a non-power-of-two width
// that is still a multiple of 64 (e.g. 192) used to pass validation and
// silently produce garbage. The prim must reject it loudly; the public
// ttnn.sort composite always pads to the next power of two, so no legal
// caller is affected.
TEST_F(SortPrimValidationFixture, RejectsNonPowerOfTwoWidth) {
    auto& device = *device_;
    const auto input = ttnn::zeros(ttnn::Shape({1, 1, 32, 192}), DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    EXPECT_THROW(
        ttnn::prim::sort(
            input,
            /*dim=*/-1,
            /*descending=*/false,
            /*stable=*/false,
            input.memory_config(),
            std::vector<std::optional<Tensor>>{}),
        std::exception);
}

// Power-of-two control: the tightened validation must keep accepting the
// widths the composite actually dispatches.
TEST_F(SortPrimValidationFixture, AcceptsPowerOfTwoWidth) {
    auto& device = *device_;
    const auto input = ttnn::zeros(ttnn::Shape({1, 1, 32, 256}), DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    EXPECT_NO_THROW(ttnn::prim::sort(
        input,
        /*dim=*/-1,
        /*descending=*/false,
        /*stable=*/false,
        input.memory_config(),
        std::vector<std::optional<Tensor>>{}));
}

}  // namespace ttnn::prim::test
