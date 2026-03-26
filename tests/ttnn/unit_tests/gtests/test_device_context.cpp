// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <exception>
#include <future>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>

#include "common_test_utils.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_context.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn_test_fixtures.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"

namespace ttnn::test {

// Creates a sub-device manager with 3 sub-devices (IDs 0, 1, 2), each with 2 cores in one row,
// loads it, and sets the stall group so get_sub_device_ids() returns {0, 1, 2}.
// Returns the SubDeviceManagerId for teardown (remove_sub_device_manager).
static tt::tt_metal::SubDeviceManagerId setup_three_sub_devices(tt::tt_metal::distributed::MeshDevice* device) {
    constexpr int cores_per_subdevice = 2;
    constexpr int num_sub_devices = 3;
    std::vector<tt::tt_metal::CoreRangeSet> core_range_sets;
    core_range_sets.reserve(num_sub_devices);
    for (int row = 0; row < num_sub_devices; ++row) {
        tt::tt_metal::CoreRange range(
            tt::tt_metal::CoreCoord(0, row), tt::tt_metal::CoreCoord(cores_per_subdevice - 1, row));
        core_range_sets.push_back(tt::tt_metal::CoreRangeSet(range));
    }
    std::vector<tt::tt_metal::SubDevice> sub_devices;
    sub_devices.reserve(num_sub_devices);
    for (int i = 0; i < num_sub_devices; ++i) {
        sub_devices.push_back(
            tt::tt_metal::SubDevice(ttsl::Span<const tt::tt_metal::CoreRangeSet>(&core_range_sets[i], 1)));
    }
    const auto id = device->create_sub_device_manager(
        ttsl::Span<const tt::tt_metal::SubDevice>(sub_devices.data(), sub_devices.size()), tt::tt_metal::DeviceAddr{0});
    device->load_sub_device_manager(id);
    const std::array<tt::tt_metal::SubDeviceId, num_sub_devices> ids = {
        tt::tt_metal::SubDeviceId{0}, tt::tt_metal::SubDeviceId{1}, tt::tt_metal::SubDeviceId{2}};
    device->set_sub_device_stall_group(ttsl::Span<const tt::tt_metal::SubDeviceId>(ids.data(), ids.size()));
    return id;
}

// Two sub-devices (IDs 0, 1), each with 2 cores in one row, for parallel execution tests.
static tt::tt_metal::SubDeviceManagerId setup_two_sub_devices(tt::tt_metal::distributed::MeshDevice* device) {
    constexpr int cores_per_subdevice = 2;
    constexpr int num_sub_devices = 2;
    std::vector<tt::tt_metal::CoreRangeSet> core_range_sets;
    core_range_sets.reserve(num_sub_devices);
    for (int row = 0; row < num_sub_devices; ++row) {
        tt::tt_metal::CoreRange range(
            tt::tt_metal::CoreCoord(0, row), tt::tt_metal::CoreCoord(cores_per_subdevice - 1, row));
        core_range_sets.push_back(tt::tt_metal::CoreRangeSet(range));
    }
    std::vector<tt::tt_metal::SubDevice> sub_devices;
    sub_devices.reserve(num_sub_devices);
    for (int i = 0; i < num_sub_devices; ++i) {
        sub_devices.push_back(
            tt::tt_metal::SubDevice(ttsl::Span<const tt::tt_metal::CoreRangeSet>(&core_range_sets[i], 1)));
    }
    const auto id = device->create_sub_device_manager(
        ttsl::Span<const tt::tt_metal::SubDevice>(sub_devices.data(), sub_devices.size()), tt::tt_metal::DeviceAddr{0});
    device->load_sub_device_manager(id);
    const std::array<tt::tt_metal::SubDeviceId, num_sub_devices> ids = {
        tt::tt_metal::SubDeviceId{0}, tt::tt_metal::SubDeviceId{1}};
    device->set_sub_device_stall_group(ttsl::Span<const tt::tt_metal::SubDeviceId>(ids.data(), ids.size()));
    return id;
}

class ExecutionContextFixture : public TTNNFixtureWithDevice {
protected:
    tt::tt_metal::SubDeviceManagerId sub_device_manager_id_;

    void SetUp() override {
        if (num_devices_ < 1) {
            GTEST_SKIP() << "No device available; skipping execution context tests.";
        }
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Sub-device managers require fast dispatch; skipping execution context tests "
                            "(TT_METAL_SLOW_DISPATCH_MODE=1).";
        }
        try {
            TTNNFixtureWithDevice::SetUp();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Device open failed (" << e.what() << "); skipping execution context tests.";
        }
        sub_device_manager_id_ = setup_three_sub_devices(device_);
    }

    void TearDown() override {
        if (device_ != nullptr) {
            device_->reset_sub_device_stall_group();
            device_->clear_loaded_sub_device_manager();
            device_->remove_sub_device_manager(sub_device_manager_id_);
        }
        TTNNFixtureWithDevice::TearDown();
    }
};

TEST_F(ExecutionContextFixture, GetCurrentSubDeviceIdDefaultsToFirst) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    ttnn::DeviceContext ctx(device);
    const auto default_id = ctx.get_current_sub_device_id();
    EXPECT_EQ(default_id, device->get_sub_device_ids().at(0));
}

TEST_F(ExecutionContextFixture, SetCurrentSubDeviceUpdatesGetCurrent) {
    auto* device = device_;
    ASSERT_NE(device, nullptr);
    ttnn::DeviceContext ctx(device);
    const auto default_id = ctx.get_current_sub_device_id();

    {
        auto guard = ctx.set_current_sub_device(tt::tt_metal::SubDeviceId{1});
        EXPECT_EQ(ttnn::DeviceContext(device).get_current_sub_device_id(), tt::tt_metal::SubDeviceId{1});
    }
    EXPECT_EQ(ttnn::DeviceContext(device).get_current_sub_device_id(), default_id);
}

// Fixture with 2 CQs and 2 subdevices for parallel execution tests.
class ParallelExecutionContextFixture : public TTNNFixtureBase {
protected:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    tt::tt_metal::SubDeviceManagerId sub_device_manager_id_;

    void SetUp() override {
        if (num_devices_ < 1) {
            GTEST_SKIP() << "No device available; skipping parallel execution context tests.";
        }
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Parallel execution requires fast dispatch; skipping "
                            "(TT_METAL_SLOW_DISPATCH_MODE=1).";
        }
        try {
            device_holder_ = ttnn::open_mesh_device(
                /*device_id=*/0,
                l1_small_size_,
                trace_region_size_,
                /*num_command_queues=*/2);
            device_ = device_holder_.get();
            sub_device_manager_id_ = setup_two_sub_devices(device_);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Device open failed (" << e.what() << "); skipping parallel execution tests.";
        }
    }

    void TearDown() override {
        if (device_ != nullptr) {
            device_->reset_sub_device_stall_group();
            device_->clear_loaded_sub_device_manager();
            device_->remove_sub_device_manager(sub_device_manager_id_);
            device_->close();
            device_ = nullptr;
            device_holder_.reset();
        }
    }
};

// Runs two streams in parallel using DeviceContext: stream 0 (subdevice 0, CQ 0) runs eltwise ops,
// stream 1 (subdevice 1, CQ 1) runs matmul. Verifies both complete and produce expected results.
TEST_F(ParallelExecutionContextFixture, ParallelExecutionTwoStreams) {
    ASSERT_NE(device_, nullptr);
    using namespace tt::tt_metal;
    const ttnn::Shape matmul_shape = ttnn::Shape{1, 1, 32, 32};
    const ttnn::Shape eltwise_shape = ttnn::Shape{1, 1, 32, 32};
    const MemoryConfig mem_cfg(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    TensorSpec matmul_spec(matmul_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
    TensorSpec eltwise_spec(eltwise_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));

    std::vector<bfloat16> a_data(matmul_shape.volume(), bfloat16(1.0f));
    std::vector<bfloat16> b_data(matmul_shape.volume(), bfloat16(1.0f));
    Tensor tensor_a = Tensor::from_vector(std::move(a_data), matmul_spec).to_device(device_);
    Tensor tensor_b = Tensor::from_vector(std::move(b_data), matmul_spec).to_device(device_);

    std::vector<bfloat16> eltwise_data(eltwise_shape.volume(), bfloat16(0.0f));  // dispatch_ops -> -32*0+128=128
    Tensor eltwise_input = Tensor::from_vector(std::move(eltwise_data), eltwise_spec).to_device(device_);

    ttnn::QueueId cq0(0);
    ttnn::QueueId cq1(1);
    constexpr tt::tt_metal::SubDeviceId sub0{0};
    constexpr tt::tt_metal::SubDeviceId sub1{1};

    Tensor matmul_result;
    auto matmul_future = std::async(std::launch::async, [&]() {
        auto cq_guard = ttnn::with_command_queue_id(cq1);
        auto sub_guard = ttnn::DeviceContext(device_).set_current_sub_device(sub1);
        matmul_result = ttnn::matmul(
            tensor_a,
            tensor_b,
            false,
            false,
            std::nullopt,
            std::nullopt,
            ttnn::operations::matmul::MatmulMultiCoreProgramConfig{});
    });

    Tensor eltwise_result;
    auto eltwise_future = std::async(std::launch::async, [&]() {
        auto cq_guard = ttnn::with_command_queue_id(cq0);
        auto sub_guard = ttnn::DeviceContext(device_).set_current_sub_device(sub0);
        eltwise_result = ttnn::test_utils::dispatch_ops_to_device(eltwise_input, cq0);
    });

    ASSERT_NO_THROW(matmul_future.get());
    ASSERT_NO_THROW(eltwise_future.get());

    tt::tt_metal::distributed::Synchronize(device_, std::nullopt);

    auto matmul_cpu = matmul_result.cpu().to_vector<bfloat16>();
    const float expected_matmul = 32.0f;  // 32x32 matmul of ones
    for (size_t i = 0; i < matmul_cpu.size(); i++) {
        EXPECT_NEAR(static_cast<float>(matmul_cpu[i]), expected_matmul, 2.0f) << "matmul idx " << i;
    }

    auto eltwise_cpu = eltwise_result.cpu().to_vector<bfloat16>();
    const float expected_eltwise = 128.0f;  // dispatch_ops_to_device: -32x+128 with x=0 -> 128
    for (size_t i = 0; i < eltwise_cpu.size(); i++) {
        EXPECT_NEAR(static_cast<float>(eltwise_cpu[i]), expected_eltwise, 2.0f) << "eltwise idx " << i;
    }
}

}  // namespace ttnn::test
