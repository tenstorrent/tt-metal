// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstdint>

#include <gtest/gtest.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/operations/data_movement/bcast_new/bcast_new.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

#include <tt_stl/fmt.hpp>
#include "/home/maxim-artemov/workspace/debug_include.hpp"

namespace ttnn::operations::data_movement::bcast::test {

template <class F>
static int64_t time_iterations_ns(int iterations, F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

class BcastDescriptorBenchmark : public TTNNFixtureWithSuiteDevice<BcastDescriptorBenchmark> {};

TEST_F(BcastDescriptorBenchmark, CorrectnessNonCached) {
    auto& device = *device_;

    std::array<uint32_t, 4> dims_a = {1, 1, 128, 128};
    ttnn::Shape shape_a(dims_a);
    std::array<uint32_t, 4> dims_b_hw = {1, 1, 32, 32};
    ttnn::Shape shape_b_hw(dims_b_hw);

    const auto a = ttnn::ones(shape_a, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto b_hw = ttnn::full(shape_b_hw, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    auto old_result = ttnn::bcast(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW);
    auto new_result = ttnn::bcast_new(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW);
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(old_result), ttnn::from_device(new_result), 0.2f, 0.02f));

    std::array<uint32_t, 4> dims_b_h = {1, 1, 32, 128};
    ttnn::Shape shape_b_h(dims_b_h);
    const auto b_h = ttnn::full(shape_b_h, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    old_result = ttnn::bcast(a, b_h, BcastOpMath::ADD, BcastOpDim::H);
    new_result = ttnn::bcast_new(a, b_h, BcastOpMath::ADD, BcastOpDim::H);
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(old_result), ttnn::from_device(new_result), 0.2f, 0.02f));

    std::array<uint32_t, 4> dims_b_w = {1, 1, 128, 32};
    ttnn::Shape shape_b_w(dims_b_w);
    const auto b_w = ttnn::full(shape_b_w, 1.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    old_result = ttnn::bcast(a, b_w, BcastOpMath::SUB, BcastOpDim::W);
    new_result = ttnn::bcast_new(a, b_w, BcastOpMath::SUB, BcastOpDim::W);
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(old_result), ttnn::from_device(new_result), 0.2f, 0.02f));
}

TEST_F(BcastDescriptorBenchmark, CorrectnessCached) {
    auto& device = *device_;

    std::array<uint32_t, 4> dims_a = {1, 1, 128, 128};
    ttnn::Shape shape_a(dims_a);
    std::array<uint32_t, 4> dims_b_hw = {1, 1, 32, 32};
    ttnn::Shape shape_b_hw(dims_b_hw);

    const auto a = ttnn::ones(shape_a, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto b_hw = ttnn::full(shape_b_hw, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    ttnn::bcast(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW);
    ttnn::bcast_new(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW);
    auto old_result = ttnn::bcast(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW);
    auto new_result = ttnn::bcast_new(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW);
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(old_result), ttnn::from_device(new_result), 0.2f, 0.02f));
}

TEST_F(BcastDescriptorBenchmark, DispatchPerformance) {
    auto& device = *device_;

    std::array<uint32_t, 4> dims_a = {1, 1, 128, 128};
    ttnn::Shape shape_a(dims_a);
    std::array<uint32_t, 4> dims_b_hw = {1, 1, 32, 32};
    ttnn::Shape shape_b_hw(dims_b_hw);
    const auto a = ttnn::ones(shape_a, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto b_hw = ttnn::full(shape_b_hw, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    constexpr int N = 10'000;

    auto t_new = time_iterations_ns(N, [&]() { (void)ttnn::bcast_new(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW); });
    auto t_old = time_iterations_ns(N, [&]() { (void)ttnn::bcast(a, b_hw, BcastOpMath::MUL, BcastOpDim::HW); });

    double overhead = (static_cast<double>(t_new) / static_cast<double>(t_old) - 1.0) * 100.0;
    std::cout << "Bcast descriptor path overhead: " << overhead << "%" << std::endl;
    EXPECT_LT(overhead, 5.0);
}

}  // namespace ttnn::operations::data_movement::bcast::test
