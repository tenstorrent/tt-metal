// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This is a standalone C++ version of run_operation_chain.py
// It runs the same operation chain to verify operation tracking and callstack generation
//
// NOTE: This binary is built by CMake (see tests/tt_metal/tools/triage/CMakeLists.txt)
// Output: build/test/tools/triage/run_operation_chain_cpp
//
// The binary is compiled WITHOUT unity build to preserve debug symbols for addr2line.
// This allows proper callstack resolution when debugging operations.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

// Include ttnn headers
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/functions.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>

using namespace tt::tt_metal;

// Step 4: Add scalar to result (this is where the hang will occur in the test)
// Use noinline attribute to ensure this function is not inlined so we can see it in the callstack
tt::tt_metal::Tensor perform_final_add(const tt::tt_metal::Tensor& input) {
    std::cout << "Step 4: result3 + 1.2" << std::endl;
    return ttnn::add(input, 1.2f);
}

// Step 3: Subtract tensor_c and then perform final add
tt::tt_metal::Tensor perform_subtract_and_add(const tt::tt_metal::Tensor& input, const tt::tt_metal::Tensor& tensor_c) {
    std::cout << "Step 3: result2 - tensor_c" << std::endl;
    auto result3 = ttnn::subtract(input, tensor_c);
    return perform_final_add(result3);
}

// Steps 1-2: Multiply operations, then call subtract and add
tt::tt_metal::Tensor perform_operation_chain(
    const tt::tt_metal::Tensor& tensor_a, const tt::tt_metal::Tensor& tensor_b, const tt::tt_metal::Tensor& tensor_c) {
    std::cout << "=== Starting operation chain ===" << std::endl;

    // Operation 1: multiply tensor_a by scalar 2.5 (matching Python: ttnn.mul(tensor_a, 2.5))
    std::cout << "Step 1: tensor_a * 2.5" << std::endl;
    auto result1 = ttnn::multiply(tensor_a, 2.5f);

    // Operation 2: multiply result1 and tensor_b (matching Python: ttnn.mul(result1, tensor_b))
    std::cout << "Step 2: result1 * tensor_b" << std::endl;
    auto result2 = ttnn::multiply(result1, tensor_b);

    // Continue with subtract and add operations
    return perform_subtract_and_add(result2, tensor_c);
}

int main() {
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

        // Perform the full operation chain
        auto final_result = perform_operation_chain(tensor_a, tensor_b, tensor_c);

        std::cout << "=== Operation chain complete ===" << std::endl;

        // Convert final result back to host for verification
        auto host_result = final_result.cpu().to_layout(Layout::ROW_MAJOR);

        // Basic verification that we got a result with correct shape
        auto result_shape = host_result.logical_shape();
        if (result_shape[0] != 1 || result_shape[1] != 1 || result_shape[2] != 32 || result_shape[3] != 64) {
            std::cerr << "ERROR: Unexpected result shape!" << std::endl;
            device->close();
            return 1;
        }

        std::cout << "=== Test completed successfully ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        device->close();
        return 1;
    }

    // Close device
    device->close();

    // Note: The actual ops.yaml file will be generated when the Inspector
    // destructor is called at program exit.

    return 0;
}
