// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <array>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "hostdevcommon/common_values.hpp"

using std::vector;
using namespace tt::tt_metal;

/*
 * Similar to loopback programming example, except run on all devices.
 * Tests that we can recover from a "bad" state.
 */

// Custom fixture for clean init test - needs multiple devices
class CleanInitFixture : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            GTEST_SKIP() << "Test not supported with slow dispatch";
        }
    }
};

TEST_F(CleanInitFixture, CleanInit) {
    auto num_devices = GetNumAvailableDevices();
    vector<tt::ChipId> ids;
    ids.reserve(num_devices);
    for (unsigned int id = 0; id < num_devices; id++) {
        ids.push_back(id);
    }

    const auto& dispatch_core_config = MetalContext::instance().rtoptions().get_dispatch_core_config();
    auto devices = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);

    for (size_t device_id = 0; device_id < num_devices; device_id++) {
        auto device = devices[device_id];
        auto& cq = device->mesh_command_queue();

        constexpr uint32_t single_tile_size = 2 * (32 * 32);
        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

        distributed::DeviceLocalBufferConfig local_l1_config{
            .page_size = dram_buffer_size, .buffer_type = BufferType::L1};

        distributed::DeviceLocalBufferConfig local_dram_config{
            .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

        distributed::ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};

        auto l1_buffer = distributed::MeshBuffer::create(global_buffer_config, local_l1_config, device.get());
        auto input_dram_buffer = distributed::MeshBuffer::create(global_buffer_config, local_dram_config, device.get());
        auto output_dram_buffer =
            distributed::MeshBuffer::create(global_buffer_config, local_dram_config, device.get());

        Program program = CreateProgram();
        auto mesh_workload = distributed::MeshWorkload();

        constexpr CoreCoord core = {0, 0};

        std::vector<uint32_t> compile_time_args;
        TensorAccessorArgs(*(input_dram_buffer->get_backing_buffer())).append_to(compile_time_args);
        TensorAccessorArgs(*(output_dram_buffer->get_backing_buffer())).append_to(compile_time_args);
        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args});

        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);

        const std::array<uint32_t, 8> runtime_args = {
            l1_buffer->address(),
            input_dram_buffer->address(),
            output_dram_buffer->address(),
            l1_buffer->size(),
            num_tiles};

        SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);
        mesh_workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
        log_info(tt::LogTest, "Started program on device {}", device_id);
        distributed::Finish(cq);
        log_info(tt::LogTest, "Finished program on device {}", device_id);

        std::vector<uint32_t> result_vec;
        distributed::ReadShard(cq, result_vec, output_dram_buffer, distributed::MeshCoordinate(0, 0));

        EXPECT_EQ(input_vec, result_vec) << "Mismatch on device " << device_id;
    }

    // Proper teardown
    for (auto& [device_id, device] : devices) {
        device->close();
    }
}
