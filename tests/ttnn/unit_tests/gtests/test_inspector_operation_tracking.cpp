// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This test runs a simple operation chain similar to tests/tt_metal/tools/triage/run_operation_chain.py
// It is used to to verify operation tracking works for C++ code

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

// Include ttnn headers
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"  // for arange and other creation ops
#include "ttnn/operations/functions.hpp"          // for random and other functions
#include "ttnn/api/ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/tensor/shape/shape.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>

#include "common_test_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::test {

TEST(InspectorOperationTracking, CppOperationChain) {
    // Set up environment for inspector
    setenv("TT_METAL_INSPECTOR_LOG_PATH", "generated/inspector", 1);

    // Clean up any existing ops.yaml files before test
    std::filesystem::path ops_file = "./generated/inspector/ops/ops.yaml";
    if (std::filesystem::exists(ops_file)) {
        std::filesystem::remove(ops_file);
    }

    // Create and open device using MeshDevice
    auto device_holder = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto device = device_holder.get();

    try {
        // Create input tensors matching Python test: shape = [32, 64]
        // Note: TTNN uses 4D shapes, so we need [1, 1, 32, 64]
        ttnn::Shape shape({1, 1, 32, 64});

        // Create random tensors (equivalent to torch.rand with seed 42)
        auto tensor_a = ttnn::random::random(shape, DataType::BFLOAT16);
        auto tensor_b = ttnn::random::random(shape, DataType::BFLOAT16);
        auto tensor_c = ttnn::random::random(shape, DataType::BFLOAT16);

        // Convert to device tensors with tile layout
        tensor_a = tensor_a.to_layout(Layout::TILE).to_device(device);
        tensor_b = tensor_b.to_layout(Layout::TILE).to_device(device);
        tensor_c = tensor_c.to_layout(Layout::TILE).to_device(device);

        std::cout << "=== Starting operation chain ===" << std::endl;

        // Operation 1: multiply tensor_a by scalar 2.5 (matching Python: ttnn.mul(tensor_a, 2.5))
        std::cout << "Step 1: tensor_a * 2.5" << std::endl;
        auto result1 = ttnn::multiply(tensor_a, 2.5f);

        // Operation 2: multiply result1 and tensor_b (matching Python: ttnn.mul(result1, tensor_b))
        std::cout << "Step 2: result1 * tensor_b" << std::endl;
        auto result2 = ttnn::multiply(result1, tensor_b);

        // Operation 3: subtract tensor_c from result2 (matching Python: ttnn.subtract(result2, tensor_c))
        std::cout << "Step 3: result2 - tensor_c" << std::endl;
        auto result3 = ttnn::subtract(result2, tensor_c);

        // Operation 4: add scalar 1.0 to result3 (matching Python: ttnn.add(result3, 1.0))
        std::cout << "Step 4: result3 + 1.0" << std::endl;
        auto final_result = ttnn::add(result3, 1.0f);

        std::cout << "=== Operation chain complete ===" << std::endl;

        // Convert final result back to host for verification
        auto host_result = final_result.cpu().to_layout(Layout::ROW_MAJOR);

        // Basic verification that we got a result with correct shape
        auto result_shape = host_result.logical_shape();
        EXPECT_EQ(result_shape[0], 1);
        EXPECT_EQ(result_shape[1], 1);
        EXPECT_EQ(result_shape[2], 32);
        EXPECT_EQ(result_shape[3], 64);

        std::cout << "=== Test completed successfully ===" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test failed with exception: " << e.what();
    }

    // Close device
    device->close();

    // Note: The actual ops.yaml file will be generated when the Inspector
    // destructor is called at program exit.
}

}  // namespace ttnn::operations::test
