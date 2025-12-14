// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/memory_utils.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
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

    ttml::utils::MemoryUsageTracker::start_capture();
    test();
    ttml::utils::MemoryUsageTracker::end_capture();

    // Get DRAM usage
    auto dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();

    size_t binary_size = 16384;          // Size of DRAM buffer used for matmul program
    size_t expected_size = binary_size;  // Allocated left over is program cache
    size_t expected_peak_size = tensor1_size + tensor2_size + result_size + expected_size;

    auto assert_dram_usage = [](const auto& dram_usage, size_t expected_size, size_t expected_peak_size) {
        EXPECT_FALSE(dram_usage.peak.empty());
        EXPECT_FALSE(dram_usage.current.empty());
        for (const auto& [dev_id, peak] : dram_usage.peak) {
            EXPECT_EQ(peak, expected_peak_size);
            EXPECT_EQ(dram_usage.current.at(dev_id), expected_size);
        }
    };
    assert_dram_usage(dram_usage, expected_size, expected_peak_size);

    // Second test with cache disabled
    device->disable_and_clear_program_cache();

    ttml::utils::MemoryUsageTracker::start_capture();
    test();
    ttml::utils::MemoryUsageTracker::end_capture();

    dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();
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

    ttml::utils::MemoryUsageTracker::start_capture();

    auto add_result = ttnn::add(tensor1, tensor2);           // (1, 1, 128, 32) + (1, 1, 128, 32) = (1, 1, 128, 32)
    auto mul_result = ttnn::multiply(tensor2, 2.0F);         // (1, 1, 128, 32) * 2.0 = (1, 1, 128, 32)
    auto matmul_result = ttnn::matmul(mul_result, tensor3);  // (1, 1, 128, 32) @ (1, 1, 32, 128) = (1, 1, 128, 128)
    auto sdpa_result = ttnn::transformer::scaled_dot_product_attention(
        q, k, v);  // (1, 6, 256, 64) @ (1, 6, 256, 64) @ (1, 6, 256, 64) = (1, 6, 256, 64)

    ttml::utils::MemoryUsageTracker::end_capture();

    size_t expected_size = 0;
    size_t expected_peak_size = 0;
    expected_size += compute_tensor_size(add_result);
    expected_size += compute_tensor_size(mul_result);
    expected_size += compute_tensor_size(matmul_result);
    expected_size += compute_tensor_size(sdpa_result);

    // tensor 2 is converted to row major + to_layout for some reason allocated additional 4096 bytes
    // TODO: Trace those extra allocations. 99% those are programs caches + intermediate tensors.
    expected_peak_size = compute_tensor_size(add_result) + compute_tensor_size(tensor2) + 4096;
    expected_peak_size += compute_tensor_size(mul_result) + 10240;
    expected_peak_size += compute_tensor_size(matmul_result) + 18432;
    expected_peak_size += compute_tensor_size(sdpa_result) + 36864;

    expected_size = expected_peak_size;

    auto dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();
    EXPECT_FALSE(dram_usage.peak.empty());
    EXPECT_FALSE(dram_usage.current.empty());
    for (const auto& [dev_id, peak] : dram_usage.peak) {
        EXPECT_EQ(peak, expected_peak_size);
        EXPECT_EQ(dram_usage.current[dev_id], expected_size);
    }

    auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();
    EXPECT_FALSE(l1_usage.peak_cb.empty());
    EXPECT_FALSE(l1_usage.current.empty());
    for (const auto& [dev_id, current] : l1_usage.current) {
        EXPECT_EQ(current, 0);
        EXPECT_EQ(l1_usage.peak_buffer[dev_id], 0);
    }
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

        ttml::utils::MemoryUsageTracker::start_capture();
        auto add_result = ttnn::add(tensor1, tensor2);
        size_t add_result_size = compute_tensor_size(add_result);
        ttml::utils::MemoryUsageTracker::end_capture();

        auto dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();
        auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();

        // DRAM usage should be 0 since we're using L1 tensors
        for (const auto& [dev_id, current] : dram_usage.current) {
            // TODO: verify that 12288 comes from program cache
            EXPECT_EQ(current, 12288);
        }

        // num_cores = 64 (256 * 256 / (32 * 32))
        // peak_cb = tile_size * sizeof(bfloat16) * num_cores * n_cb (cb0, cb1, cb_out)
        size_t expected_peak_cb = 2048 * 2 * 64 * 3;
        for (const auto& [dev_id, peak_cb] : l1_usage.peak_cb) {
            EXPECT_EQ(peak_cb, expected_peak_cb);
        }

        // current L1 should be zero after operation completes
        size_t expected_current = add_result_size;
        for (const auto& [dev_id, current] : l1_usage.current) {
            EXPECT_EQ(current, expected_current);
        }

        // peak_buffer should be volume of tensors (one result tensor in L1)
        size_t expected_peak_buffer = expected_current;
        for (const auto& [dev_id, peak_buffer] : l1_usage.peak_buffer) {
            EXPECT_EQ(peak_buffer, expected_peak_buffer);
        }
    }

    // Second capture: two 128x128 tensors added
    {
        auto shape = ttnn::Shape({128, 128});
        std::vector<float> data(128 * 128, 2.0F);

        // Create tensors in DRAM first, then move to L1
        auto tensor1_dram = ttml::core::from_vector(data, shape, device, ttnn::TILE_LAYOUT);
        auto tensor2_dram = ttml::core::from_vector(data, shape, device, ttnn::TILE_LAYOUT);
        auto tensor1 = ttnn::to_memory_config(tensor1_dram, ttnn::L1_MEMORY_CONFIG);
        auto tensor2 = ttnn::to_memory_config(tensor2_dram, ttnn::L1_MEMORY_CONFIG);

        ttml::utils::MemoryUsageTracker::start_capture();
        auto add_result = ttnn::add(tensor1, tensor2);
        size_t add_result_size = compute_tensor_size(add_result);
        ttml::utils::MemoryUsageTracker::end_capture();

        auto dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();
        auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();

        // DRAM usage should be 0 since we're using L1 tensors
        for (const auto& [dev_id, current] : dram_usage.current) {
            EXPECT_EQ(current, 0);
        }

        // num_cores = 16 (128 * 128 / (32 * 32))
        // peak_cb = tile_size * sizeof(bfloat16) * num_cores * n_cb (cb0, cb1, cb_out)
        size_t expected_peak_cb = 2048 * 2 * 16 * 3;
        for (const auto& [dev_id, peak_cb] : l1_usage.peak_cb) {
            EXPECT_EQ(peak_cb, expected_peak_cb);
        }

        // peak_buffer should be volume of tensors (one result tensor in L1)
        size_t expected_peak_buffer = add_result_size;
        for (const auto& [dev_id, peak_buffer] : l1_usage.peak_buffer) {
            EXPECT_EQ(peak_buffer, expected_peak_buffer);
        }

        // current L1 should be zero after operation completes
        size_t expected_current = expected_peak_buffer;
        for (const auto& [dev_id, current] : l1_usage.current) {
            EXPECT_EQ(current, expected_current);
        }
    }
}
