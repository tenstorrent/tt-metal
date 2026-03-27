// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/tensor/shape/shape.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

class ProfilerNoOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ProfilerNoOpTest, ProfilerNoOpTest_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 187U;

    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> input_tensor =
        ttml::test_utils::make_uniform_xarray<float>(std::array<std::size_t, 4>{N, C, H, W}, seed, -10.0F, 10.0F);

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::profiler_no_op(input, "identifier");

    // NOTE: ProfilerNoOp does not change the input, so we just check that the operation completed successfully.
}
