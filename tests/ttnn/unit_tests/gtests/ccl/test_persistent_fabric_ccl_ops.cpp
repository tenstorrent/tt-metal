
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <umd/device/types/arch.hpp>
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"

TEST(CclAsyncOp, ReduceScatterSmall_PersistentFabric) {
    const size_t dim = 3;
    constexpr auto layout = Layout::TILE;
    // DEVICES setup
    constexpr size_t test_expected_num_devices = 4;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info(tt::LogTest, "This test can only be run on T3000 devices");
        return;
    }
    MeshFabric1DFixture test_fixture(tt::tt_fabric::FabricConfig::FABRIC_1D);

    // build a line of devices
    const size_t num_devices = test_expected_num_devices;
    const ttnn::Shape input_shape({1, 1, 32, 32 * num_devices});
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    // INPUT TENSOR setup

    // Replicate the tensor across (1, num_devices) submesh.
    const Tensor input_mesh_tensor = ttnn::distributed::distribute_tensor(
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to_layout(layout),
        *ttnn::distributed::create_mesh_mapper(
            *test_fixture.mesh_device_,
            ttnn::distributed::MeshMapperConfig{
                .placements =
                    {ttnn::distributed::MeshMapperConfig::Replicate{},
                     ttnn::distributed::MeshMapperConfig::Replicate{}},
                .mesh_shape_override = MeshShape{1, num_devices}}),
        *test_fixture.mesh_device_);

    auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim, 1);
    // auto output_tensor = ttnn::reduce_scatter(input_mesh_tensor, dim); //Issue 31845, should be able to replace
    // previous line with this

    // wait for op completion
    log_info(tt::LogTest, "Waiting for Op finish");
    tt_metal::distributed::Finish(test_fixture.mesh_device_->mesh_command_queue(), {{SubDeviceId(0)}});

    log_info(tt::LogTest, "Finished");
}
