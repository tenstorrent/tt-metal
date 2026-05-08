// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/dropout/dropout.hpp"
#include "ttnn/operations/experimental/dropout_new/dropout_new.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::experimental::test {

class DropoutDescriptorBenchmark : public TTNNFixtureWithSuiteDevice<DropoutDescriptorBenchmark> {};

namespace {

constexpr float kProb = 0.2f;
constexpr float kScale = 1.0f;
constexpr uint32_t kSeed = 1337;
constexpr bool kUsePerDeviceSeed = false;

constexpr int kDispatchIterations = 10000;

Tensor make_input(tt::tt_metal::distributed::MeshDevice& device) {
    const std::array<uint32_t, 4> dims = {1, 1, 256, 256};
    const Shape shape(dims);
    return ttnn::ones(shape, DataType::BFLOAT16, Layout::TILE, device);
}

Tensor call_old(const Tensor& input) {
    return ttnn::experimental::dropout(input, kProb, kScale, kSeed, kUsePerDeviceSeed);
}

Tensor call_new(const Tensor& input) {
    return ttnn::experimental::dropout_new(input, kProb, kScale, kSeed, kUsePerDeviceSeed);
}

bool outputs_match(const Tensor& lhs, const Tensor& rhs) {
    return ttnn::allclose<::bfloat16>(ttnn::from_device(lhs), ttnn::from_device(rhs), 0.0f, 0.0f);
}

template <typename Fn>
uint64_t measure_ns(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

}  // namespace

TEST_F(DropoutDescriptorBenchmark, CorrectnessNonCached) {
    auto& device = *device_;
    const auto input = make_input(device);

    const auto old_output = call_old(input);
    const auto new_output = call_new(input);

    ASSERT_TRUE(outputs_match(old_output, new_output));
}

TEST_F(DropoutDescriptorBenchmark, CorrectnessCached) {
    auto& device = *device_;
    const auto input = make_input(device);

    (void)call_old(input);
    (void)call_new(input);

    const auto old_output = call_old(input);
    const auto new_output = call_new(input);

    ASSERT_TRUE(outputs_match(old_output, new_output));
}

TEST_F(DropoutDescriptorBenchmark, DispatchPerformance) {
    auto& device = *device_;
    const auto input = make_input(device);

    // Warm up both paths once before timing.
    (void)call_old(input);
    (void)call_new(input);

    const auto new_time_ns = measure_ns([&]() {
        for (int i = 0; i < kDispatchIterations; ++i) {
            (void)call_new(input);
        }
    });

    const auto old_time_ns = measure_ns([&]() {
        for (int i = 0; i < kDispatchIterations; ++i) {
            (void)call_old(input);
        }
    });

    const double overhead_pct = (static_cast<double>(new_time_ns) / static_cast<double>(old_time_ns) - 1.0) * 100.0;
    std::cout << "Dropout descriptor dispatch overhead (%): " << overhead_pct << std::endl;

    EXPECT_LT(overhead_pct, 3.0);
}

}  // namespace ttnn::operations::experimental::test
