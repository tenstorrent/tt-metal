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
    MeshDevice* device;
};

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
