// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/constants.hpp>
#include <functional>
#include <gtest/gtest.h>

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

class StorageTypeValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_id = 0;
        device_owner = MeshDevice::create_unit_mesh(device_id);
        device = device_owner.get();
    }

    void TearDown() override { device_owner.reset(); }

    std::shared_ptr<MeshDevice> device_owner;
    MeshDevice* device{};
};

// Helper function to create host tensors and call binary operation
// This should trigger our storage type validation and throw a runtime error
void call_binary_with_host_tensors(bool use_legacy) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});

    // Create host tensors using the proper method (like in copy_and_move test)
    auto random_data = ttnn::random::random(shape, DataType::BFLOAT16);
    auto host_tensor_a = random_data.cpu();  // Move to host
    auto host_tensor_b = random_data.cpu();  // Move to host

    // Verify these are host tensors
    TT_FATAL(
        host_tensor_a.storage_type() == tt::tt_metal::StorageType::HOST, "Input tensor A should be HOST storage type");
    TT_FATAL(
        host_tensor_b.storage_type() == tt::tt_metal::StorageType::HOST, "Input tensor B should be HOST storage type");

    // This should fail with storage type validation error and throw a runtime error
    auto result = ttnn::operations::binary::BinaryOperation<ttnn::operations::binary::BinaryOpType::ADD>::invoke(
        ttnn::DefaultQueueId,
        host_tensor_a,
        host_tensor_b,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        use_legacy);

    // If we get here, the validation failed (which is bad)
    TT_FATAL(false, "Expected storage type validation to fail, but operation succeeded");
}

TEST_F(StorageTypeValidationTest, BinaryOperationHostTensorValidation) {
    // Test that binary operation (use_legacy=true) throws runtime error when given host tensors
    // This is the expected behavior - our storage type validation should throw an exception
    EXPECT_THROW(call_binary_with_host_tensors(true), std::runtime_error);
}

TEST_F(StorageTypeValidationTest, BinaryNgOperationHostTensorValidation) {
    // Test that binary_ng operation (use_legacy=false) throws runtime error when given host tensors
    // This is the expected behavior - our storage type validation should throw an exception
    EXPECT_THROW(call_binary_with_host_tensors(false), std::runtime_error);
}

TEST_F(StorageTypeValidationTest, DeviceTensorsWorkCorrectly) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});

    // Create tensors in TILE layout first, then move to device (should work)
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);

    // Verify these are device tensors
    EXPECT_EQ(input_tensor_a.storage_type(), tt::tt_metal::StorageType::DEVICE);
    EXPECT_EQ(input_tensor_b.storage_type(), tt::tt_metal::StorageType::DEVICE);

    // This should work without crashing
    EXPECT_NO_THROW({
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
    });
}

TEST_F(StorageTypeValidationTest, BinaryNgOperationDeviceTensorsWorkCorrectly) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});

    // Create tensors in TILE layout first, then move to device (should work)
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16).to_layout(Layout::TILE).to_device(device);

    // Verify these are device tensors
    EXPECT_EQ(input_tensor_a.storage_type(), tt::tt_metal::StorageType::DEVICE);
    EXPECT_EQ(input_tensor_b.storage_type(), tt::tt_metal::StorageType::DEVICE);

    // This should work without crashing
    EXPECT_NO_THROW({
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
    });
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
