// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/concat_new/concat_new.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::test {

class ConcatDescriptorBenchmark : public TTNNFixtureWithSuiteDevice<ConcatDescriptorBenchmark> {};

namespace {

std::vector<Tensor> create_inputs(ttnn::MeshDevice& device) {
    const Shape lhs_shape({1, 1, 64, 64});
    const Shape rhs_shape({1, 1, 64, 96});
    return {
        ttnn::ones(lhs_shape, DataType::BFLOAT16, Layout::TILE, device, ttnn::DRAM_MEMORY_CONFIG),
        ttnn::full(rhs_shape, 0.5f, DataType::BFLOAT16, Layout::TILE, device, ttnn::DRAM_MEMORY_CONFIG)};
}

Tensor run_old(ttnn::MeshDevice& device) {
    auto inputs = create_inputs(device);
    return ttnn::concat(inputs, -1, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, 1, std::nullopt);
}

Tensor run_new(ttnn::MeshDevice& device) {
    auto inputs = create_inputs(device);
    return ttnn::concat_new(inputs, -1, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, 1, std::nullopt);
}

}  // namespace

TEST_F(ConcatDescriptorBenchmark, CorrectnessNonCached) {
    auto& device = *device_;
    auto old_result = run_old(device);
    auto new_result = run_new(device);
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(old_result), ttnn::from_device(new_result), 1e-2f, 1e-2f));
}

TEST_F(ConcatDescriptorBenchmark, CorrectnessCached) {
    auto& device = *device_;
    run_old(device);
    run_new(device);

    auto old_result = run_old(device);
    auto new_result = run_new(device);
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(old_result), ttnn::from_device(new_result), 1e-2f, 1e-2f));
}

TEST_F(ConcatDescriptorBenchmark, DispatchPerformance) {
    auto& device = *device_;
    constexpr int N = 10000;

    auto start_new = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        auto out = run_new(device);
        (void)out;
    }
    auto end_new = std::chrono::steady_clock::now();

    auto start_old = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        auto out = run_old(device);
        (void)out;
    }
    auto end_old = std::chrono::steady_clock::now();

    const auto new_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_new - start_new).count();
    const auto old_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_old - start_old).count();
    const double overhead = (static_cast<double>(new_ns) / static_cast<double>(old_ns) - 1.0) * 100.0;

    std::cout << "Concat descriptor overhead: " << overhead << "%" << std::endl;
    EXPECT_LT(overhead, 3.0);
}

}  // namespace ttnn::operations::data_movement::test
