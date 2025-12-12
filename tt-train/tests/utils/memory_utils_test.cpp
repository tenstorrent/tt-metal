// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/memory_utils.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

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

TEST_F(MemoryUtilsTest, DRAMUsageBasic) {
    auto* device = &ttml::autograd::ctx().get_device();

    size_t expected_size = 0;

    std::vector<float> data1(64 * 128, 1.0F);
    std::vector<float> data2(128 * 64, 2.0F);

    // Start memory capture
    ttml::utils::MemoryUsageTracker::start_capture();

    // Create a few tensors in DRAM (default memory location)
    auto shape1 = ttnn::Shape({64, 128});
    auto shape2 = ttnn::Shape({128, 64});

    auto tensor1 = ttml::core::from_vector(data1, shape1, device, ttnn::TILE_LAYOUT);
    auto tensor2 = ttml::core::from_vector(data2, shape2, device, ttnn::TILE_LAYOUT);
    expected_size += compute_tensor_size(tensor1);
    expected_size += compute_tensor_size(tensor2);

    // Perform an operation that creates a new tensor (matmul)
    // auto result = ttnn::matmul(tensor1, tensor2);
    // auto result = ttnn::add(tensor1, tensor2);
    // expected_size += compute_tensor_size(result);

    // End capture
    ttml::utils::MemoryUsageTracker::end_capture();

    // Get DRAM usage
    auto dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();

    // Should have at least one device entry
    EXPECT_FALSE(dram_usage.peak.empty());
    EXPECT_FALSE(dram_usage.current.empty());

    std::cout << "expected size: " << expected_size << std::endl;
    // Verify peak is at least as large as current
    for (const auto& [dev_id, peak] : dram_usage.peak) {
        std::cout << "Device " << dev_id << " Peak DRAM: " << peak << " bytes" << std::endl;
        EXPECT_EQ(peak, expected_size);
    }
    for (const auto& [dev_id, current] : dram_usage.current) {
        EXPECT_EQ(current, expected_size);
    }
}

// TEST_F(MemoryUtilsTest, DRAMUsageMultipleTensors) {
//     auto* device = &ttml::autograd::ctx().get_device();

//     ttml::utils::MemoryUsageTracker::start_capture();

//     // Create multiple tensors of various sizes
//     auto shape1 = ttnn::Shape({1, 1, 32, 32});
//     auto shape2 = ttnn::Shape({1, 1, 64, 64});
//     auto shape3 = ttnn::Shape({1, 1, 128, 128});

//     std::vector<float> data1(32 * 32, 1.0F);
//     std::vector<float> data2(64 * 64, 2.0F);
//     std::vector<float> data3(128 * 128, 3.0F);

//     auto tensor1 = ttml::core::from_vector(data1, shape1, device);
//     auto tensor2 = ttml::core::from_vector(data2, shape2, device);
//     auto tensor3 = ttml::core::from_vector(data3, shape3, device);

//     // Do some operations
//     auto add_result = ttnn::add(tensor2, tensor2);
//     auto mul_result = ttnn::multiply(tensor3, 2.0F);

//     ttml::utils::MemoryUsageTracker::end_capture();

//     auto dram_usage = ttml::utils::MemoryUsageTracker::get_DRAM_usage();

//     EXPECT_FALSE(dram_usage.peak.empty());

//     for (const auto& [dev_id, peak] : dram_usage.peak) {
//         fmt::print(
//             "DRAMUsageMultipleTensors - Device {}: Peak DRAM {} bytes, Current DRAM {} bytes\n",
//             dev_id,
//             peak,
//             dram_usage.current[dev_id]);
//         // Peak should be positive
//         EXPECT_GT(peak, 0);
//     }
// }

// TEST_F(MemoryUtilsTest, L1UsageWithL1Tensors) {
//     auto* device = &ttml::autograd::ctx().get_device();

//     ttml::utils::MemoryUsageTracker::start_capture();

//     // Create tensors and move them to L1
//     auto shape = ttnn::Shape({1, 1, 64, 64});
//     std::vector<float> data(64 * 64, 1.0F);

//     auto tensor_dram = ttml::core::from_vector(data, shape, device);
//     auto tensor_l1 = ttml::ttnn_fixed::to_l1_interleaved(tensor_dram);

//     // Create another L1 tensor
//     auto tensor_dram2 = ttml::core::from_vector(data, shape, device);
//     auto tensor_l1_2 = ttml::ttnn_fixed::to_l1_interleaved(tensor_dram2);

//     ttml::utils::MemoryUsageTracker::end_capture();

//     auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();

//     for (const auto& [dev_id, peak_buffer] : l1_usage.peak_buffer) {
//         fmt::print(
//             "L1UsageWithL1Tensors - Device {}: Peak L1 Buffer {} bytes, Current L1 {} bytes\n",
//             dev_id,
//             peak_buffer,
//             l1_usage.current[dev_id]);
//     }

//     // Should have some L1 buffer usage from the L1 tensors
//     EXPECT_FALSE(l1_usage.peak_buffer.empty());
// }

// TEST_F(MemoryUtilsTest, L1UsageWithAddOp) {
//     auto* device = &ttml::autograd::ctx().get_device();

//     ttml::utils::MemoryUsageTracker::start_capture();

//     // Create tensors
//     auto shape = ttnn::Shape({1, 1, 64, 64});
//     std::vector<float> data(64 * 64, 1.0F);

//     auto tensor1 = ttml::core::from_vector(data, shape, device);
//     auto tensor2 = ttml::core::from_vector(data, shape, device);

//     // Perform add operation - this uses circular buffers
//     auto add_result = ttnn::add(tensor1, tensor2);

//     ttml::utils::MemoryUsageTracker::end_capture();

//     auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();

//     fmt::print("\n=== L1 Usage with Add Operation ===\n");
//     for (const auto& [dev_id, peak_cb] : l1_usage.peak_cb) {
//         fmt::print(
//             "Device {}: Peak CB {} bytes, Peak Buffer {} bytes, Peak Total {} bytes\n",
//             dev_id,
//             peak_cb,
//             l1_usage.peak_buffer[dev_id],
//             l1_usage.peak_total[dev_id]);
//     }

//     // Add operation should use some circular buffer memory
//     // The actual value will be printed for the user to add to assertions
//     EXPECT_FALSE(l1_usage.peak_cb.empty());
// }

// TEST_F(MemoryUtilsTest, L1UsageWithMatmulOp) {
//     auto* device = &ttml::autograd::ctx().get_device();

//     ttml::utils::MemoryUsageTracker::start_capture();

//     // Create tensors for matmul
//     auto shape1 = ttnn::Shape({1, 1, 64, 128});
//     auto shape2 = ttnn::Shape({1, 1, 128, 64});

//     std::vector<float> data1(64 * 128, 1.0F);
//     std::vector<float> data2(128 * 64, 1.0F);

//     auto tensor1 = ttml::core::from_vector(data1, shape1, device);
//     auto tensor2 = ttml::core::from_vector(data2, shape2, device);

//     // Perform matmul operation - this typically uses more CB memory than add
//     auto matmul_result = ttnn::matmul(tensor1, tensor2);

//     ttml::utils::MemoryUsageTracker::end_capture();

//     auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();

//     fmt::print("\n=== L1 Usage with Matmul Operation ===\n");
//     for (const auto& [dev_id, peak_cb] : l1_usage.peak_cb) {
//         fmt::print(
//             "Device {}: Peak CB {} bytes, Peak Buffer {} bytes, Peak Total {} bytes\n",
//             dev_id,
//             peak_cb,
//             l1_usage.peak_buffer[dev_id],
//             l1_usage.peak_total[dev_id]);
//     }

//     // Matmul should use circular buffers
//     EXPECT_FALSE(l1_usage.peak_cb.empty());
// }

// TEST_F(MemoryUtilsTest, L1UsageWithAddAndMatmul) {
//     auto* device = &ttml::autograd::ctx().get_device();

//     ttml::utils::MemoryUsageTracker::start_capture();

//     // Create tensors
//     auto shape1 = ttnn::Shape({1, 1, 64, 128});
//     auto shape2 = ttnn::Shape({1, 1, 128, 64});

//     std::vector<float> data1(64 * 128, 1.0F);
//     std::vector<float> data2(128 * 64, 1.0F);

//     auto tensor1 = ttml::core::from_vector(data1, shape1, device);
//     auto tensor2 = ttml::core::from_vector(data2, shape2, device);

//     // Perform matmul
//     auto matmul_result = ttnn::matmul(tensor1, tensor2);

//     // Perform add on the result
//     auto add_result = ttnn::add(matmul_result, matmul_result);

//     ttml::utils::MemoryUsageTracker::end_capture();

//     auto l1_usage = ttml::utils::MemoryUsageTracker::get_L1_usage();

//     fmt::print("\n=== L1 Usage with Add and Matmul Operations ===\n");
//     for (const auto& [dev_id, peak_cb] : l1_usage.peak_cb) {
//         fmt::print(
//             "Device {}: Peak CB {} bytes, Peak Buffer {} bytes, Peak Total {} bytes\n",
//             dev_id,
//             peak_cb,
//             l1_usage.peak_buffer[dev_id],
//             l1_usage.peak_total[dev_id]);
//     }

//     // Should use circular buffers
//     EXPECT_FALSE(l1_usage.peak_cb.empty());
// }

// TEST_F(MemoryUtilsTest, PrintMemoryUsage) {
//     auto* device = &ttml::autograd::ctx().get_device();

//     ttml::utils::MemoryUsageTracker::start_capture();

//     // Create some tensors and operations
//     auto shape = ttnn::Shape({1, 1, 128, 128});
//     std::vector<float> data(128 * 128, 1.0F);

//     auto tensor1 = ttml::core::from_vector(data, shape, device);
//     auto tensor2 = ttml::core::from_vector(data, shape, device);
//     auto tensor_l1 = ttml::ttnn_fixed::to_l1_interleaved(tensor1);

//     auto matmul_result = ttnn::matmul(tensor1, tensor2);
//     auto add_result = ttnn::add(matmul_result, tensor2);

//     ttml::utils::MemoryUsageTracker::end_capture();

//     // Test the print function
//     fmt::print("\n=== Full Memory Usage Summary ===\n");
//     ttml::utils::MemoryUsageTracker::print_memory_usage();
// }
