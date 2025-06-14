
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "umd/device/types/arch.h"

TEST(CclAsyncOp, ReduceScatterSmall_PersistentFabric) {
    const size_t dim = 3;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    // DEVICES setup
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    constexpr size_t test_expected_num_devices = 4;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }
    MeshFabric1DFixture test_fixture(tt::tt_metal::FabricConfig::FABRIC_1D);
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(1, 1)),
        view.get_device(MeshCoordinate(1, 2)),
        view.get_device(MeshCoordinate(0, 2))};
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);
    const ttnn::Shape input_shape({1, 1, 32, 32 * num_devices});
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    // INPUT TENSOR setup
    size_t page_size = tile_size(DataFormat::Float16);
    std::vector<Tensor> device_input_tensors;
    for (size_t i = 0; i < num_devices; i++) {
        // host_input_tensors.push_back(ttnn::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f) ,
        // {input_shape[0],input_shape[1],input_shape[2],input_shape[3]}, layout).to_device(devices[i]));
        auto t =
            ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to_layout(layout);
        device_input_tensors.push_back(t);
    }
    // Need to make it a mesh tensor for use with the op
    const Tensor input_mesh_tensor = ttnn::distributed::aggregate_as_tensor(device_input_tensors, AllGatherTensor{})
                                         .to_device(test_fixture.mesh_device_.get());

    std::optional<SubdeviceInfo> subdevice_managers = create_worker_subdevices(devices);

    GlobalSemaphore from_remote_multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore(
        test_fixture.mesh_device_.get(),
        devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                            // initial value
        tt::tt_metal::BufferType::L1  // buffer type
    );

    GlobalSemaphore to_remote_multi_device_global_semaphore = ttnn::global_semaphore::create_global_semaphore(
        test_fixture.mesh_device_.get(),
        devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
        0,                            // initial value
        tt::tt_metal::BufferType::L1  // buffer type
    );

    auto output_tensor = ttnn::operations::experimental::ccl::reduce_scatter(
        input_mesh_tensor,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType::Sum,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology::Linear,
        num_links,
        subdevice_managers->worker_subdevice_id.at(devices[0]->id()));

    // wait for op completion
    log_info(tt::LogTest, "Waiting for Op finish");
    std::ranges::for_each(devices, [&](IDevice* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->worker_subdevice_id.at(d->id())});
    });

    log_info(tt::LogTest, "Finished");
}

TEST(CclAsyncOp, AllGather_PersistentFabric_Dim3_Links1_Shape1_1_32_128) {
    run_all_gather_with_persistent_fabric(3, 1, ttnn::Shape({1, 1, 32, 128}));
}
TEST(CclAsyncOp, AllGather_PersistentFabric_Dim3_Links1_Shape1_1_32_8192) {
    run_all_gather_with_persistent_fabric(3, 1, ttnn::Shape({1, 1, 32, 8192}));
}
// Mesh device setup seems to not provide the correct configuration for multi-link? To be investigated
TEST(CclAsyncOp, DISABLED_AllGather_PersistentFabric_Dim3_Links2_Shape1_1_32_128) {
    run_all_gather_with_persistent_fabric(3, 2, ttnn::Shape({1, 1, 32, 128}));
}
// Mesh device setup seems to not provide the correct configuration for multi-link? To be investigated
TEST(CclAsyncOp, DISABLED_AllGather_PersistentFabric_Dim3_Links2_Shape1_1_32_8192) {
    run_all_gather_with_persistent_fabric(3, 2, ttnn::Shape({1, 1, 32, 8192}));
}

TEST(CclAsyncOp, RingAllGather_PersistentFabric_Dim3_Links1_Shape1_256_32_8192) {
    run_ring_all_gather_with_persistent_fabric(3, 1, ttnn::Shape({1, 256, 32, 8192}));
}
