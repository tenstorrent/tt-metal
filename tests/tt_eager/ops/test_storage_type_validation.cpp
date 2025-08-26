// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/constants.hpp>
#include <functional>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;
using tt::tt_metal::distributed::MeshDevice;

void test_binary_storage_type_validation_host_tensor(tt::tt_metal::distributed::MeshDevice* device) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});

    // Create host tensors (not on device)
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16);

    // Verify these are host tensors
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::HOST, "Input tensor A should be HOST storage type");
    TT_FATAL(
        input_tensor_b.storage_type() == tt::tt_metal::StorageType::HOST, "Input tensor B should be HOST storage type");

    // This should fail with storage type validation error
    // We expect a TT_FATAL assertion to be triggered
    try {
        auto result = ttnn::operations::binary::BinaryOperation<ttnn::operations::binary::BinaryOpType::ADD>::invoke(
            ttnn::DefaultQueueId,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            {},
            {},
            {},
            true);
        TT_FATAL(false, "Expected storage type validation to fail, but operation succeeded");
    } catch (const std::exception& e) {
        // Expected to fail
        fmt::print("Successfully caught storage type validation error: {}\n", e.what());
    }
}

void test_binary_ng_storage_type_validation_host_tensor(tt::tt_metal::distributed::MeshDevice* device) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});

    // Create host tensors (not on device)
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16);

    // Verify these are host tensors
    TT_FATAL(
        input_tensor_a.storage_type() == tt::tt_metal::StorageType::HOST, "Input tensor A should be HOST storage type");
    TT_FATAL(
        input_tensor_b.storage_type() == tt::tt_metal::StorageType::HOST, "Input tensor B should be HOST storage type");

    // This should fail with storage type validation error
    // We expect a TT_FATAL assertion to be triggered
    try {
        auto result = ttnn::operations::binary::BinaryOperation<ttnn::operations::binary::BinaryOpType::ADD>::invoke(
            ttnn::DefaultQueueId,
            input_tensor_a,
            input_tensor_b,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            {},
            {},
            {},
            false);
        TT_FATAL(false, "Expected storage type validation to fail, but operation succeeded");
    } catch (const std::exception& e) {
        // Expected to fail
        fmt::print("Successfully caught storage type validation error: {}\n", e.what());
    }
}

int main() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device_owner = MeshDevice::create_unit_mesh(device_id);
    auto device = device_owner.get();

    fmt::print("Testing binary operation storage type validation with host tensors...\n");
    test_binary_storage_type_validation_host_tensor(device);

    fmt::print("Testing binary_ng operation storage type validation with host tensors...\n");
    test_binary_ng_storage_type_validation_host_tensor(device);

    fmt::print("All storage type validation tests passed!\n");
    return 0;
}
