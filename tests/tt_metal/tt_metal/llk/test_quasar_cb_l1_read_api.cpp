// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "host_api.hpp"
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/host_api/temp_quasar_api.hpp"
#include "llk_device_fixture.hpp"
#include "tt_metal.hpp"

namespace tt::tt_metal {

namespace {

static constexpr CoreCoord WORKER_CORE = {0, 0};

using DataT = std::uint32_t;
static constexpr auto DATA_FORMAT = DataFormat::UInt32;

static constexpr std::size_t CB_PAGE_SIZE = 16;

static constexpr DataT VAL0 = 0xA5A5A5A5u;
static constexpr DataT VAL1 = 0x11111111u;
static const std::vector<DataT> EXPECTED_RESULT = {VAL0, VAL1, VAL0};

static constexpr std::size_t RESULT_BUFFER_PAGE_SIZE = CB_PAGE_SIZE;
static constexpr std::size_t RESULT_BUFFER_SIZE = RESULT_BUFFER_PAGE_SIZE;
static constexpr auto RESULT_BUFFER_TYPE = BufferType::L1;

std::shared_ptr<distributed::MeshBuffer> create_result_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = RESULT_BUFFER_PAGE_SIZE,
        .buffer_type = RESULT_BUFFER_TYPE,
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = RESULT_BUFFER_SIZE,
    };
    auto result_buffer = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    std::vector<DataT> init_data(RESULT_BUFFER_SIZE / sizeof(DataT), 0);
    distributed::WriteShard(
        mesh_device->mesh_command_queue(), result_buffer, init_data, distributed::MeshCoordinate(0, 0));
    return result_buffer;
}

}  // namespace

// Validates ckernel::read_tile_value and ckernel::get_tile_address on Quasar (cb_api.h).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarCbL1ReadApi) {
    auto mesh_device = devices_.at(0);
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    const uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(
        program_,
        WORKER_CORE,
        experimental::dfb::DataflowBufferConfig{
            .entry_size = static_cast<uint32_t>(CB_PAGE_SIZE),
            .num_entries = 1,
            .data_format = DATA_FORMAT,
        });

    auto reader_kernel = experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_cb_l1_read_api_reader.cpp",
        WORKER_CORE,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {dfb_id},
        });

    auto compute_kernel = experimental::quasar::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/circular_buffer/quasar_cb_l1_read_api_compute.cpp",
        WORKER_CORE,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {dfb_id},
        });

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program_, dfb_id, reader_kernel, compute_kernel);

    auto result_buffer = create_result_buffer(mesh_device);
    SetRuntimeArgs(program_, compute_kernel, WORKER_CORE, {result_buffer->address()});

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    std::vector<DataT> host_buffer;
    const auto expected_result_size = EXPECTED_RESULT.size() * sizeof(DataT);
    detail::ReadFromDeviceL1(device, WORKER_CORE, result_buffer->address(), expected_result_size, host_buffer);

    EXPECT_EQ(host_buffer, EXPECTED_RESULT);
}

}  // namespace tt::tt_metal
