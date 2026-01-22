// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/memory_utils.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/types.hpp"

class MemoryUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

size_t compute_tensor_size(const ttnn::Tensor& tensor) {
    auto physical_shape = tensor.padded_shape();
    return physical_shape.volume() * tensor.element_size();
}

TEST_F(MemoryUtilsTest, DRAMUsageMatmulInScope) {
    auto* device = &ttml::autograd::ctx().get_device();

    std::vector<float> data1(64 * 128, 1.0F);
    std::vector<float> data2(128 * 64, 2.0F);

    size_t tensor1_size = 0;
    size_t tensor2_size = 0;
    size_t result_size = 0;
    auto test = [&]() {
        // Create a few tensors in DRAM (default memory location)
        auto shape1 = ttnn::Shape({64, 128});
        auto shape2 = ttnn::Shape({128, 64});

        ttnn::TensorSpec spec1(
            shape1,
            ttnn::TensorLayout(
                ttnn::DataType::BFLOAT16,
                ttnn::PageConfig(ttnn::Layout::TILE),
                ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
        ttnn::TensorSpec spec2(
            shape2,
            ttnn::TensorLayout(
                ttnn::DataType::BFLOAT16,
                ttnn::PageConfig(ttnn::Layout::TILE),
                ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
        auto tensor1 = ttnn::Tensor::from_vector(data1, spec1, device);
        auto tensor2 = ttnn::Tensor::from_vector(data2, spec2, device);

        tensor1_size = compute_tensor_size(tensor1);
        tensor2_size = compute_tensor_size(tensor2);

        // Perform an operation that creates a new tensor (matmul)
        auto result = ttnn::matmul(tensor1, tensor2);
        result_size = compute_tensor_size(result);
    };

    // First test with cache enabled (enabled by default)

    auto guard1 = ttml::utils::MemoryUsageTracker::begin_capture();
    test();
    ttml::utils::MemoryUsageTracker::end_capture();

    // Get DRAM usage
    auto dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage();

    size_t binary_size = 16384;          // Size of DRAM buffer used for matmul program
    size_t expected_size = binary_size;  // Allocated left over is program cache
    size_t expected_peak_size = tensor1_size + tensor2_size + result_size + expected_size;

    auto assert_dram_usage = [](const auto& dram_usage, size_t expected_size, size_t expected_peak_size) {
        EXPECT_EQ(dram_usage.peak, expected_peak_size);
        EXPECT_EQ(dram_usage.total_allocations - dram_usage.total_deallocations, expected_size);
    };
    assert_dram_usage(dram_usage, expected_size, expected_peak_size);

    // Second test with cache disabled
    device->disable_and_clear_program_cache();

    auto guard2 = ttml::utils::MemoryUsageTracker::begin_capture();
    test();
    ttml::utils::MemoryUsageTracker::end_capture();

    dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage();
    expected_size = 0;
    expected_peak_size = tensor1_size + tensor2_size + result_size + binary_size;  // Binary size is still allocated
    assert_dram_usage(dram_usage, expected_size, expected_peak_size);
}

TEST_F(MemoryUtilsTest, DRAMUsageMultipleOperations) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Create multiple tensors of various sizes
    auto shape1 = ttnn::Shape({1, 1, 128, 32});
    auto shape2 = ttnn::Shape({1, 1, 128, 32});
    auto shape3 = ttnn::Shape({1, 1, 32, 128});
    auto shape_kqv = ttnn::Shape({1, 6, 256, 64});

    std::vector<float> data1(1 * 1 * 128 * 32, 1.0F);
    std::vector<float> data2(1 * 1 * 128 * 32, 2.0F);
    std::vector<float> data3(1 * 1 * 32 * 128, 3.0F);

    std::vector<float> data_kqv(1 * 6 * 256 * 64, 4.0F);

    ttnn::TensorSpec spec1(
        shape1,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
    ttnn::TensorSpec spec2(
        shape2,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
    ttnn::TensorSpec spec3(
        shape3,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
    ttnn::TensorSpec spec_kqv(
        shape_kqv,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
    auto tensor1 = ttnn::Tensor::from_vector(data1, spec1, device);
    auto tensor2 = ttnn::Tensor::from_vector(data2, spec2, device);
    auto tensor3 = ttnn::Tensor::from_vector(data3, spec3, device);
    auto q = ttnn::Tensor::from_vector(data_kqv, spec_kqv, device);
    auto k = ttnn::Tensor::from_vector(data_kqv, spec_kqv, device);
    auto v = ttnn::Tensor::from_vector(data_kqv, spec_kqv, device);
    auto q_tensor = ttml::autograd::create_tensor(q);
    auto k_tensor = ttml::autograd::create_tensor(k);
    auto v_tensor = ttml::autograd::create_tensor(v);

    auto guard = ttml::utils::MemoryUsageTracker::begin_capture();

    auto add_result = ttnn::add(tensor1, tensor2);           // (1, 1, 128, 32) + (1, 1, 128, 32) = (1, 1, 128, 32)
    auto mul_result = ttnn::multiply(tensor2, 2.0F);         // (1, 1, 128, 32) * 2.0 = (1, 1, 128, 32)
    auto matmul_result = ttnn::matmul(mul_result, tensor3);  // (1, 1, 128, 32) @ (1, 1, 32, 128) = (1, 1, 128, 128)
    auto sdpa_result = ttml::ops::scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor);  // (1, 6, 256, 64) @ (1, 6, 256, 64) @ (1, 6, 256, 64) = (1, 6, 256, 64)

    ttml::utils::MemoryUsageTracker::end_capture();

    size_t expected_size = 0;
    size_t expected_peak_size = 0;

    // tensor 2 is converted to row major + to_layout for some reason allocated additional 4096 bytes
    // TODO: Trace those extra allocations. 99% those are programs caches + intermediate tensors.
    expected_peak_size = compute_tensor_size(add_result) + compute_tensor_size(tensor2) + 4096;
    expected_peak_size += compute_tensor_size(mul_result) + 10240;
    expected_peak_size += compute_tensor_size(matmul_result) + 18432;
    expected_peak_size +=
        compute_tensor_size(sdpa_result->get_value()) + 1830912;  // All the intermediate tensors / activations

    expected_size = expected_peak_size - 983040;  // Some intermediates are deallocated

    auto dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage();
    EXPECT_EQ(dram_usage.peak, expected_peak_size);
    EXPECT_EQ(dram_usage.total_allocations - dram_usage.total_deallocations, expected_size);

    auto l1_usage = ttml::utils::MemoryUsageTracker::get_l1_usage();
    EXPECT_EQ(l1_usage.peak_l1, 0);
}

TEST_F(MemoryUtilsTest, L1Usage) {
    auto* device = &ttml::autograd::ctx().get_device();

    // First capture: two 256x256 tensors added
    {
        auto shape = ttnn::Shape({256, 256});
        std::vector<float> data(256 * 256, 1.0F);

        // Create tensors in DRAM first, then move to L1
        ttnn::TensorSpec spec1(
            shape,
            ttnn::TensorLayout(
                ttnn::DataType::BFLOAT16,
                ttnn::PageConfig(ttnn::Layout::TILE),
                ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
        ttnn::TensorSpec spec2(
            shape,
            ttnn::TensorLayout(
                ttnn::DataType::BFLOAT16,
                ttnn::PageConfig(ttnn::Layout::TILE),
                ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
        auto tensor1_dram = ttnn::Tensor::from_vector(data, spec1, device);
        auto tensor2_dram = ttnn::Tensor::from_vector(data, spec2, device);
        auto tensor1 = ttnn::to_memory_config(tensor1_dram, ttnn::L1_MEMORY_CONFIG);
        auto tensor2 = ttnn::to_memory_config(tensor2_dram, ttnn::L1_MEMORY_CONFIG);

        auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
        auto add_result = ttnn::add(tensor1, tensor2);
        size_t add_result_size = compute_tensor_size(add_result);
        ttml::utils::MemoryUsageTracker::end_capture();

        auto dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage();
        auto l1_usage = ttml::utils::MemoryUsageTracker::get_l1_usage();

        // TODO: verify that 12288 comes from program cache
        EXPECT_EQ(dram_usage.total_allocations, 12288);

        // peak_cb = tile_size * sizeof(bfloat16) * n_cb (cb0, cb1, cb_out)
        size_t expected_peak_cb = 2048 * 2 * 3;
        EXPECT_EQ(l1_usage.peak_cb, expected_peak_cb);

        // Output tensor is in L1
        EXPECT_EQ(l1_usage.peak_l1, add_result_size);

        // peak_buffer should be volume of tensors (one result tensor in L1)
        size_t expected_peak_buffer = add_result_size + expected_peak_cb;
        EXPECT_EQ(l1_usage.peak_total, expected_peak_buffer);
    }

    // Second capture: two 128x128 tensors added with width sharding
    {
        auto shape = ttnn::Shape({128, 128});
        std::vector<float> data(128 * 128, 2.0F);

        // Create width-sharded memory config
        // 128x128 in tiles = 4x4 tiles (each tile is 32x32)
        // Width sharding: distribute across width dimension (4 cores)
        uint32_t num_cores = 4;  // Sharding across 4 cores for width
        auto core_range = ttnn::CoreRangeSet({ttnn::CoreRange({0, 0}, {num_cores - 1, 0})});

        // Shard shape: each core gets 4 tiles height x 1 tile width = 128x32 elements
        auto shard_spec = tt::tt_metal::ShardSpec(core_range, {128, 32}, tt::tt_metal::ShardOrientation::ROW_MAJOR);
        auto sharded_memory_config =
            ttnn::MemoryConfig{ttnn::TensorMemoryLayout::WIDTH_SHARDED, ttnn::BufferType::L1, shard_spec};

        // Create tensors in DRAM first, then move to L1 with width sharding
        auto tensor1_dram = ttml::core::from_vector(data, shape, device, ttnn::TILE_LAYOUT);
        auto tensor2_dram = ttml::core::from_vector(data, shape, device, ttnn::TILE_LAYOUT);
        auto tensor1 = ttnn::to_memory_config(tensor1_dram, sharded_memory_config);
        auto tensor2 = ttnn::to_memory_config(tensor2_dram, sharded_memory_config);

        auto guard = ttml::utils::MemoryUsageTracker::begin_capture();
        auto add_result = ttnn::add(tensor1, tensor2);
        size_t add_result_size =
            compute_tensor_size(add_result) / num_cores;  // Each core gets 1/num_cores of the result
        ttml::utils::MemoryUsageTracker::end_capture();

        auto dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage();
        auto l1_usage = ttml::utils::MemoryUsageTracker::get_l1_usage();

        // DRAM usage from cache miss
        EXPECT_EQ(dram_usage.total_allocations, 10240);

        size_t expected_peak_cb = 0;  // CBs are not allocated since add uses sharded inputs as CBs
        EXPECT_EQ(l1_usage.peak_cb, expected_peak_cb);

        // Output tensor is in L1
        EXPECT_EQ(l1_usage.peak_l1, add_result_size);

        // peak_total should be the sum of peak_l1 and peak_cb
        size_t expected_peak_total = add_result_size + expected_peak_cb;
        EXPECT_EQ(l1_usage.peak_total, expected_peak_total);
    }
}

TEST_F(MemoryUtilsTest, SnapshotFeature) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->disable_and_clear_program_cache();

    // Prepare data for different operations
    std::vector<float> data_small(64 * 64, 1.0F);
    std::vector<float> data_medium(128 * 64, 2.0F);
    std::vector<float> data_large(256 * 256, 3.0F);

    auto guard = ttml::utils::MemoryUsageTracker::begin_capture();

    // Snapshot 1: Simple add operation with small DRAM tensors
    auto shape1 = ttnn::Shape({64, 64});
    ttnn::TensorSpec spec1(
        shape1,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));

    auto tensor1 = ttnn::Tensor::from_vector(data_small, spec1, device);
    auto tensor2 = ttnn::Tensor::from_vector(data_small, spec1, device);
    auto result1 = ttnn::add(tensor1, tensor2);

    ttml::utils::MemoryUsageTracker::snapshot("add_operation");

    // Snapshot 2: Matmul operation with DRAM tensors
    auto shape2a = ttnn::Shape({128, 64});
    auto shape2b = ttnn::Shape({64, 128});

    ttnn::TensorSpec spec2a(
        shape2a,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));
    ttnn::TensorSpec spec2b(
        shape2b,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));

    auto tensor3 = ttnn::Tensor::from_vector(data_medium, spec2a, device);
    auto tensor4 = ttnn::Tensor::from_vector(data_medium, spec2b, device);
    auto result2 = ttnn::matmul(tensor3, tensor4);

    ttml::utils::MemoryUsageTracker::snapshot("matmul_operation");

    // Snapshot 3: L1 operation
    auto shape3 = ttnn::Shape({256, 256});
    ttnn::TensorSpec spec3(
        shape3,
        ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::Layout::TILE),
            ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM)));

    auto tensor5_dram = ttnn::Tensor::from_vector(data_large, spec3, device);
    auto tensor6_dram = ttnn::Tensor::from_vector(data_large, spec3, device);

    // Move to L1 for operation
    auto tensor5 = ttnn::to_memory_config(tensor5_dram, ttnn::L1_MEMORY_CONFIG);
    auto tensor6 = ttnn::to_memory_config(tensor6_dram, ttnn::L1_MEMORY_CONFIG);
    auto result3 = ttnn::multiply(tensor5, tensor6);

    ttml::utils::MemoryUsageTracker::end_capture("multiply_l1_operation");

    // Verify that all three snapshots were captured
    auto trace_names = ttml::utils::MemoryUsageTracker::get_trace_names();
    ASSERT_EQ(trace_names.size(), 3);
    EXPECT_EQ(trace_names[0], "add_operation");
    EXPECT_EQ(trace_names[1], "matmul_operation");
    EXPECT_EQ(trace_names[2], "multiply_l1_operation");

    // Get all DRAM and L1 usage
    auto all_dram_usage = ttml::utils::MemoryUsageTracker::get_dram_usage_all();
    auto all_l1_usage = ttml::utils::MemoryUsageTracker::get_l1_usage_all();

    ASSERT_EQ(all_dram_usage.size(), 3);
    ASSERT_EQ(all_l1_usage.size(), 3);

    // Verify individual snapshots have captured memory usage with exact values
    // Snapshot 1: Add operation
    auto dram_usage_1 = ttml::utils::MemoryUsageTracker::get_dram_usage("add_operation");
    // 2 inputs + output have size of 64*64*sizeof(bfloat16) + 12288 bytes of program cache
    // 36864 = 3 * (64 * 64) * sizeof(bfloat16) + 12288
    EXPECT_EQ(dram_usage_1.peak, 36864);
    EXPECT_EQ(dram_usage_1.total_allocations, 36864);
    EXPECT_EQ(dram_usage_1.total_deallocations, 12288);

    // Snapshot 2: Matmul operation
    auto dram_usage_2 = ttml::utils::MemoryUsageTracker::get_dram_usage("matmul_operation");
    EXPECT_EQ(dram_usage_2.peak, 86016);
    EXPECT_EQ(dram_usage_2.total_allocations, 86016);
    EXPECT_EQ(dram_usage_2.total_deallocations, 20480);

    // Snapshot 3: L1 multiply operation
    auto dram_usage_3 = ttml::utils::MemoryUsageTracker::get_dram_usage("multiply_l1_operation");
    auto l1_usage_3 = ttml::utils::MemoryUsageTracker::get_l1_usage("multiply_l1_operation");
    EXPECT_EQ(dram_usage_3.peak, 274432);
    // Total DRAM allocations = (256 * 256 * sizeof(bfloat16) * 2) /*DRAM inputs*/ + 20480 /*program cache*/
    EXPECT_EQ(dram_usage_3.total_allocations, 282624);
    EXPECT_EQ(dram_usage_3.total_deallocations, 20480);
    // peak_l1 = (256 * 256 * sizeof(bfloat16)) * 2 /*DRAM inputs*/ + (256 * 256 * sizeof(bfloat16)) /*L1 output*/
    EXPECT_EQ(l1_usage_3.peak_l1, 393216);

    // Verify get_dram_usage_all and get_l1_usage_all return correct pairs
    EXPECT_EQ(all_dram_usage[0].first, "add_operation");
    EXPECT_EQ(all_dram_usage[1].first, "matmul_operation");
    EXPECT_EQ(all_dram_usage[2].first, "multiply_l1_operation");

    EXPECT_EQ(all_l1_usage[0].first, "add_operation");
    EXPECT_EQ(all_l1_usage[1].first, "matmul_operation");
    EXPECT_EQ(all_l1_usage[2].first, "multiply_l1_operation");
}
