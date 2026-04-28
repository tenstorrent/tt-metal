// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/gather/gather.hpp"
#include "ttnn/operations/data_movement/gather_new/gather_new.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::test {

class GatherDescriptorBenchmark : public TTNNFixtureWithSuiteDevice<GatherDescriptorBenchmark> {
protected:
    void run_gather_correctness(bool small_wt_path) {
        ttnn::random::seed(0);
        auto& mesh = *device_;

        ttnn::Shape input_shape;
        ttnn::Shape index_shape;
        if (small_wt_path) {
            input_shape = ttnn::Shape{8, 8, 8, 8};
            index_shape = ttnn::Shape{8, 8, 8, 8};
        } else {
            const uint32_t wt = 65;
            input_shape = ttnn::Shape{1, 1, 32, wt * tt::constants::TILE_WIDTH};
            index_shape = ttnn::Shape{1, 1, 32, (wt - 1) * tt::constants::TILE_WIDTH};
        }

        const uint32_t gather_dim_size = input_shape[3];
        auto input_host = ttnn::random::random(input_shape, DataType::BFLOAT16, Layout::TILE).to_device(&mesh);
        auto index_host = ttnn::random::uniform(0u, gather_dim_size - 1, index_shape, Layout::TILE).to_device(&mesh);

        auto out_old = ttnn::gather(input_host, -1, index_host, false, std::nullopt);
        auto out_new = ttnn::gather_new(input_host, -1, index_host, false, std::nullopt);

        ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(out_old), ttnn::from_device(out_new)));
    }

    template <typename Fn>
    static int64_t time_microseconds(int iterations, Fn&& fn) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            fn();
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};

TEST_F(GatherDescriptorBenchmark, CorrectnessNonCached_SmallWt) { run_gather_correctness(true); }

TEST_F(GatherDescriptorBenchmark, CorrectnessNonCached_LargeWt) { run_gather_correctness(false); }

TEST_F(GatherDescriptorBenchmark, CorrectnessCached) {
    ttnn::random::seed(1);
    auto& mesh = *device_;

    const ttnn::Shape input_shape{8, 8, 8, 8};
    const ttnn::Shape index_shape{8, 8, 8, 8};
    constexpr uint32_t gather_dim_size = 8;
    auto input_host = ttnn::random::random(input_shape, DataType::BFLOAT16, Layout::TILE).to_device(&mesh);
    auto index_host = ttnn::random::uniform(0u, gather_dim_size - 1, index_shape, Layout::TILE).to_device(&mesh);

    ttnn::gather(input_host, -1, index_host, false, std::nullopt);
    ttnn::gather_new(input_host, -1, index_host, false, std::nullopt);

    auto out_old = ttnn::gather(input_host, -1, index_host, false, std::nullopt);
    auto out_new = ttnn::gather_new(input_host, -1, index_host, false, std::nullopt);

    ASSERT_TRUE(ttnn::allclose<::bfloat16>(ttnn::from_device(out_old), ttnn::from_device(out_new)));
}

TEST_F(GatherDescriptorBenchmark, DispatchPerformance) {
    constexpr int kIterations = 100'000;
    ttnn::random::seed(2);
    auto& mesh = *device_;

    const ttnn::Shape input_shape{8, 8, 8, 8};
    const ttnn::Shape index_shape{8, 8, 8, 8};
    constexpr uint32_t gather_dim_size = 8;
    auto input_host = ttnn::random::random(input_shape, DataType::BFLOAT16, Layout::TILE).to_device(&mesh);
    auto index_host = ttnn::random::uniform(0u, gather_dim_size - 1, index_shape, Layout::TILE).to_device(&mesh);

    const int64_t t_new =
        time_microseconds(kIterations, [&]() { ttnn::gather_new(input_host, -1, index_host, false, std::nullopt); });
    const int64_t t_old =
        time_microseconds(kIterations, [&]() { ttnn::gather(input_host, -1, index_host, false, std::nullopt); });

    const double overhead = (static_cast<double>(t_new) / static_cast<double>(t_old) - 1.0) * 100.0;
    std::cout << "Gather descriptor dispatch overhead: " << overhead << "%" << std::endl;
    EXPECT_LT(overhead, 5.0);
}

}  // namespace ttnn::operations::data_movement::test
