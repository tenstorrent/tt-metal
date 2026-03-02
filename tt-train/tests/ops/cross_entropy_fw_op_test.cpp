
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cassert>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

class CrossEntropyForwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

xt::xarray<float> calculate_cross_entropy_loss(const xt::xarray<float>& input, const xt::xarray<uint32_t>& target) {
    const uint32_t N = target.shape(0);
    const uint32_t C = 1U;
    const uint32_t H = target.shape(1);
    const uint32_t W = 1U;
    xt::xarray<float> target_inputs = xt::zeros<float>({N, C, H, W});

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            size_t class_index = target(n, h);
            target_inputs(n, 0, h, 0) = input(n, 0, h, class_index);
        }
    }

    xt::xarray<float> max_input = xt::amax(input, -1, xt::keep_dims);
    xt::xarray<float> shifted_input = input - max_input;
    xt::xarray<float> log_exp_sum_test = xt::log(xt::sum(xt::exp(shifted_input), -1, xt::keep_dims));
    xt::xarray<float> result = -target_inputs + max_input + log_exp_sum_test;
    return result;
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Small_Forward) {
    using namespace ttml;

    const uint32_t N = 1, H = 1;

    xt::xarray<float> input_tensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 1U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Negetive_Values) {
    using namespace ttml;

    const uint32_t N = 1, H = 2;

    xt::xarray<float> input_tensor = {{{{-100.F, -101.F, -102.F, -103.F}, {-5.01F, -5.02F, -0.3F, -7.F}}}};
    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});
    target_tensor(0, 0) = 0;
    target_tensor(0, 1) = 2U;
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Batch) {
    using namespace ttml;

    const uint32_t N = 2U, C = 1U, H = 91U, W = 157U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Batch) {
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 1017U, W = 1018U;
    const auto shape = ttsl::SmallVector<uint32_t>{N, C, H, W};

    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Large_Forward) {
    using namespace ttml;

    const uint32_t N = 1U, C = 1U, H = 1U, W = 65536U;
    const auto shape = ttsl::SmallVector<size_t>{N, C, H, W};

    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, NIGHTLY_CrossEntropyForward_Huge_Forward) {
    auto board = tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0);
    if (board == tt::BoardType::P100 || board == tt::BoardType::P150) {
        GTEST_SKIP() << "Skipping on P100/P150 boards";
    }
    using namespace ttml;

    const uint32_t N = 64U, C = 1U, H = 32U, W = 128000U;
    const auto shape = ttsl::SmallVector<size_t>{N, C, H, W};

    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto input = core::from_xtensor(input_tensor, &autograd::ctx().get_device());

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_tensor, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = ttml::metal::cross_entropy_fw(input, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    // Check if the result is close to the expected result
    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_DRAM_Sharded) {
    auto board = tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0);
    if (board == tt::BoardType::P100 || board == tt::BoardType::P150) {
        GTEST_SKIP() << "Skipping on P100/P150 boards";
    }
    using namespace ttml;

    // Large shape for performance testing using ND sharding
    // Shape: (1, 1, 384, 128000) - 12 tile-rows × 4000 tile-columns
    // ND shard_shape: [1, 1, 32, 128] = 1 tile-row × 4 tiles per shard = 8KB (fits in L1!)
    // This allows fine-grained sharding where each shard is small
    const uint32_t N = 1U, C = 1U, H = 384U, W = 128000U;

    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t true_class = class_dist(gen);
            target_tensor(n, h) = true_class;
        }
    }

    auto* device = &autograd::ctx().get_device();

    // Get DRAM grid for core range
    auto dram_grid = device->dram_grid_size();
    tt::tt_metal::CoreRangeSet dram_cores(tt::tt_metal::CoreRange({0, 0}, {dram_grid.x - 1, dram_grid.y - 1}));

    // ND Shard spec: [1, 1, 32, 128] = 1 tile-row × 4 tiles = 8KB per shard
    // Tiles within each shard are contiguous - can read 4 tiles per NOC call!
    tt::tt_metal::Shape shard_shape({1, 1, 32, 128});
    tt::tt_metal::NdShardSpec nd_shard_spec{shard_shape, dram_cores, tt::tt_metal::ShardOrientation::ROW_MAJOR};
    tt::tt_metal::MemoryConfig dram_nd_sharded_config{tt::tt_metal::BufferType::DRAM, nd_shard_spec};

    std::cout << "Shape: (" << N << ", " << C << ", " << H << ", " << W << ")" << std::endl;
    std::cout << "ND Shard shape: [1, 1, 32, 128] = 1 tile-row × 4 tiles per shard" << std::endl;
    std::cout << "DRAM cores: " << dram_grid.x << " x " << dram_grid.y << std::endl;

    // Create TensorSpec with ND sharding for device tensor
    tt::tt_metal::TensorLayout sharded_layout(
        tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), dram_nd_sharded_config);
    tt::tt_metal::TensorSpec sharded_spec(tt::tt_metal::Shape({N, C, H, W}), sharded_layout);

    // Create host tensor spec (interleaved) for from_vector
    tt::tt_metal::MemoryConfig host_mem_config{};  // Default interleaved
    tt::tt_metal::TensorLayout host_layout(
        tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), host_mem_config);
    tt::tt_metal::TensorSpec host_spec(tt::tt_metal::Shape({N, C, H, W}), host_layout);

    // Create host tensor and move to device with ND sharding
    std::vector<float> input_data(input_tensor.data(), input_tensor.data() + input_tensor.size());
    auto host_tensor = ttnn::Tensor::from_vector(input_data, host_spec);
    auto input_sharded = host_tensor.to_device(device, dram_nd_sharded_config);

    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_tensor, device, ttnn::Layout::ROW_MAJOR);

    // Run the operation with DRAM ND-sharded input
    auto result = ttml::metal::cross_entropy_fw(input_sharded, target);

    auto expected_result = calculate_cross_entropy_loss(input_tensor, target_tensor);

    auto result_xtensor = core::to_xtensor(result);
    assert((result_xtensor.shape() == expected_result.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 3e-2F, 1e-2F));
}

TEST_F(CrossEntropyForwardTest, CrossEntropyForward_Perf_Comparison) {
    auto board = tt::umd::Cluster::create_cluster_descriptor()->get_board_type(0);
    if (board == tt::BoardType::P100 || board == tt::BoardType::P150) {
        GTEST_SKIP() << "Skipping on P100/P150 boards";
    }
    using namespace ttml;

    // Large shape for performance testing with ND sharding
    const uint32_t N = 1U, C = 1U, H = 2048U, W = 128000U;
    const int num_iterations = 10;

    std::mt19937 gen(42);
    xt::xarray<float> input_tensor = xt::empty<float>({N, C, H, W});
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    ttml::core::parallel_generate(
        std::span{input_tensor.data(), input_tensor.size()},
        []() { return std::uniform_real_distribution<float>(-10.0F, 10.0F); },
        seed);
    xt::xarray<uint32_t> target_tensor = xt::zeros<uint32_t>({N, H});

    std::uniform_int_distribution<uint32_t> class_dist(0, W - 1);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t h = 0; h < H; ++h) {
            target_tensor(n, h) = class_dist(gen);
        }
    }

    auto* device = &autograd::ctx().get_device();

    // Create interleaved tensor
    auto input_interleaved = core::from_xtensor(input_tensor, device);
    auto target = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(target_tensor, device, ttnn::Layout::ROW_MAJOR);

    // Setup ND sharded DRAM config
    auto dram_grid = device->dram_grid_size();
    tt::tt_metal::CoreRangeSet dram_cores(tt::tt_metal::CoreRange({0, 0}, {dram_grid.x - 1, dram_grid.y - 1}));

    // ND Shard spec: [1, 1, 32, 128] = 1 tile-row × 4 tiles = 8KB per shard
    tt::tt_metal::Shape shard_shape({1, 1, 32, 128});
    tt::tt_metal::NdShardSpec nd_shard_spec{shard_shape, dram_cores, tt::tt_metal::ShardOrientation::ROW_MAJOR};
    tt::tt_metal::MemoryConfig dram_nd_sharded_config{tt::tt_metal::BufferType::DRAM, nd_shard_spec};

    // Create sharded tensor spec - must embed memory config in TensorLayout
    tt::tt_metal::TensorLayout sharded_layout(
        tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), dram_nd_sharded_config);
    tt::tt_metal::TensorSpec sharded_spec(tt::tt_metal::Shape({N, C, H, W}), sharded_layout);

    // Create ND-sharded tensor - use the sharded spec for both from_vector and to_device
    std::vector<float> input_data(input_tensor.data(), input_tensor.data() + input_tensor.size());
    auto host_tensor = ttnn::Tensor::from_vector(input_data, sharded_spec);
    auto input_sharded = host_tensor.to_device(device, sharded_spec.memory_config());

    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Shape: (" << N << ", " << C << ", " << H << ", " << W << ")" << std::endl;
    std::cout << "ND Shard: [1, 1, 32, 128] = 1 tile-row × 4 tiles per shard" << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;

    // Warmup
    ttml::metal::cross_entropy_fw(input_interleaved, target);
    ttml::metal::cross_entropy_fw(input_sharded, target);
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Benchmark DRAM ND-SHARDED
    auto start_sharded = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto result = ttml::metal::cross_entropy_fw(input_sharded, target);
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    auto end_sharded = std::chrono::high_resolution_clock::now();
    auto duration_sharded = std::chrono::duration_cast<std::chrono::microseconds>(end_sharded - start_sharded).count();

    // Benchmark INTERLEAVED
    auto start_interleaved = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto result = ttml::metal::cross_entropy_fw(input_interleaved, target);
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    auto end_interleaved = std::chrono::high_resolution_clock::now();
    auto duration_interleaved =
        std::chrono::duration_cast<std::chrono::microseconds>(end_interleaved - start_interleaved).count();

    double avg_interleaved_ms = static_cast<double>(duration_interleaved) / num_iterations / 1000.0;
    double avg_sharded_ms = static_cast<double>(duration_sharded) / num_iterations / 1000.0;
    double speedup = avg_interleaved_ms / avg_sharded_ms;

    std::cout << "\nResults:" << std::endl;
    std::cout << "  INTERLEAVED: " << avg_interleaved_ms << " ms/iter" << std::endl;
    std::cout << "  DRAM SHARDED: " << avg_sharded_ms << " ms/iter" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;

    // Test should pass regardless of performance
    EXPECT_TRUE(true);
}
