// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn_test_fixtures.hpp"

#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp"

#include <vector>

namespace ttnn {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
global_semaphore::MultiDeviceGlobalSemaphore create_global_semaphore(
    const std::map<chip_id_t, tt::tt_metal::IDevice*>& devs) {
    std::vector<IDevice*> devices;
    for (const auto& dev : devs) {
        devices.push_back(dev.second);
    }
    return global_semaphore::create_global_semaphore_with_same_address(
        devices,
        devices.at(0)->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,
        tt::tt_metal::BufferType::L1,
        10);
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(MultiDeviceN300Fixture, AllGather) {
    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec, devs[dev_idx]));
    }
    auto all_gathered = ttnn::all_gather(tensors, 0);
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(MultiDeviceN300Fixture, AllGatherAsync) {
    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(dev_idx)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec, devs[dev_idx]));
    }
    auto semaphore = CMAKE_UNIQUE_NAMESPACE::create_global_semaphore(devs);
    auto all_gathered = ttnn::experimental::all_gather_async(tensors, 0, semaphore);
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        auto data = all_gathered[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(i / tensor_spec.logical_shape().volume());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(MultiDeviceN300Fixture, ReduceScatter) {
    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec, devs[dev_idx]));
    }
    auto reduced = ttnn::reduce_scatter(tensors, 3, operations::reduction::ReduceType::Sum);
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(devs.size());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

TEST_F(MultiDeviceN300Fixture, AllReduce) {
    std::vector<ttnn::Tensor> tensors;
    TensorSpec tensor_spec(
        ttnn::Shape({1, 8, 1024, 768}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        std::vector<bfloat16> data(tensor_spec.logical_shape().volume(), bfloat16(static_cast<float>(1)));
        tensors.push_back(Tensor::from_vector(std::move(data), tensor_spec, devs[dev_idx]));
    }
    auto reduced = ttnn::experimental::all_reduce(tensors, operations::reduction::ReduceType::Sum);
    for (int dev_idx = 0; dev_idx < devs.size(); dev_idx++) {
        auto data = reduced[dev_idx].to_vector<bfloat16>();
        for (int i = 0; i < data.size(); i++) {
            float expected = static_cast<float>(devs.size());
            EXPECT_EQ(data[i].to_float(), expected);
        }
    }
}

}  // namespace ttnn
