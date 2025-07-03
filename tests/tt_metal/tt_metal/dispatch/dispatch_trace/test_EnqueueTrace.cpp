// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stdlib.h>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "dispatch_test_utils.hpp"
#include "env_lib.hpp"
#include "gtest/gtest.h"
#include "hostdevcommon/kernel_structs.h"
#include "trace/trace_buffer.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "multi_command_queue_fixture.hpp"
#include "random_program_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/common/scoped_timer.hpp"
#include "umd/device/tt_core_coordinates.h"
#include "distributed/mesh_trace.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;

Program create_simple_unary_program(Buffer& input, Buffer& output) {
    Program program = CreateProgram();
    IDevice* device = input.device();
    CoreCoord worker = {0, 0};
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sfpu_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        worker,
        ComputeConfig{
            .math_approx_mode = true,
            .compile_args = {1, 1},
            .defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}});

    CircularBufferConfig input_cb_config = CircularBufferConfig(2048, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                                               .set_page_size(tt::CBIndex::c_0, 2048);

    CoreRange core_range({0, 0});
    CreateCircularBuffer(program, core_range, input_cb_config);
    std::shared_ptr<RuntimeArgs> writer_runtime_args = std::make_shared<RuntimeArgs>();
    std::shared_ptr<RuntimeArgs> reader_runtime_args = std::make_shared<RuntimeArgs>();

    *writer_runtime_args = {
        &output,
        (uint32_t)0,
        output.num_pages()
    };

    *reader_runtime_args = {
        &input,
        (uint32_t)0,
        input.num_pages()
    };

    SetRuntimeArgs(device, detail::GetKernel(program, writer_kernel), worker, writer_runtime_args);
    SetRuntimeArgs(device, detail::GetKernel(program, reader_kernel), worker, reader_runtime_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                                                .set_page_size(tt::CBIndex::c_16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    return program;
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

constexpr bool kBlocking = true;
constexpr bool kNonBlocking = false;
vector<bool> blocking_flags = {kBlocking, kNonBlocking};

TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueOneProgramTrace) {
    CreateDevices(2048);
    auto mesh_device = this->devices_[0];

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    auto& mesh_command_queue = mesh_device->mesh_command_queue(0);
    auto& data_movement_queue = mesh_device->mesh_command_queue(1);

    // Create program and add to workload
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program simple_program =
        create_simple_unary_program(*input->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
    distributed::AddProgramToMeshWorkload(workload, std::move(simple_program), device_range);

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Eager mode
    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());

    distributed::EnqueueWriteMeshBuffer(data_movement_queue, input, input_data, true);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);
    distributed::ReadShard(data_movement_queue, eager_output_data, output, distributed::MeshCoordinate{0, 0}, true);
    // distributed::EnqueueReadMeshBuffer(data_movement_queue, eager_output_data, output, true);

    // Trace mode
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    distributed::EnqueueWriteMeshBuffer(data_movement_queue, input, input_data, true);

    auto tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);

    distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), tid, true);
    distributed::ReadShard(data_movement_queue, trace_output_data, output, distributed::MeshCoordinate{0, 0}, true);
    // distributed::EnqueueReadMeshBuffer(data_movement_queue, trace_output_data, output, true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(mesh_device.get(), tid);  //?
}

TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueOneProgramTraceLoops) {
    CreateDevices(4096);
    auto mesh_device = this->devices_[0];

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    auto& mesh_command_queue = mesh_device->mesh_command_queue(0);
    auto& data_movement_queue = mesh_device->mesh_command_queue(1);

    // Create program and add to workload
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program simple_program =
        create_simple_unary_program(*input->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
    distributed::AddProgramToMeshWorkload(workload, std::move(simple_program), device_range);

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = 10;
    vector<vector<uint32_t>> trace_outputs;

    for (auto i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    // Compile
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);

    // Trace mode execution
    distributed::MeshTraceId trace_id;
    bool trace_captured = false;
    for (auto i = 0; i < num_loops; i++) {
        distributed::EnqueueWriteMeshBuffer(data_movement_queue, input, input_data, true);

        if (not trace_captured) {
            trace_id = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), trace_id);
            trace_captured = true;
        }

        distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), trace_id, false);
        distributed::ReadShard(data_movement_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, true);

        // Expect same output across all loops
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }

    // Done
    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(mesh_device.get(), trace_id);
}

TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueOneProgramTraceBenchmark) {
    CreateDevices(6144);
    auto mesh_device = this->devices_[0];

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    constexpr bool kBlocking = true;
    constexpr bool kNonBlocking = false;
    vector<bool> blocking_flags = {kBlocking, kNonBlocking};

    // Single Q for data and commands
    // Keep this queue in passthrough mode for now
    auto& mesh_command_queue = mesh_device->mesh_command_queue(0);

    // Create program and add to workload
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program simple_program =
        create_simple_unary_program(*input->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
    distributed::AddProgramToMeshWorkload(workload, std::move(simple_program), device_range);

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = 10;
    vector<vector<uint32_t>> trace_outputs;

    for (auto i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    // Eager mode
    vector<uint32_t> expected_output_data;
    vector<uint32_t> eager_output_data;
    expected_output_data.resize(input_data.size());
    eager_output_data.resize(input_data.size());

    // Warm up and use the eager blocking run as the expected output
    distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kBlocking);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, kBlocking);
    distributed::ReadShard(
        mesh_command_queue, expected_output_data, output, distributed::MeshCoordinate{0, 0}, kBlocking);
    distributed::Finish(mesh_command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            tt::ScopedTimer timer(mode + " loop " + std::to_string(i));
            distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, blocking);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, blocking);
            distributed::ReadShard(
                mesh_command_queue, eager_output_data, output, distributed::MeshCoordinate{0, 0}, blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            distributed::Finish(mesh_command_queue);
        }
        EXPECT_TRUE(eager_output_data == expected_output_data);
    }

    // Capture trace on a trace queue
    auto tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        tt::ScopedTimer timer("Trace loop " + std::to_string(i));
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kNonBlocking);
        distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), tid, kNonBlocking);
        distributed::ReadShard(
            mesh_command_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, kNonBlocking);
    }
    distributed::Finish(mesh_command_queue);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
    distributed::ReleaseTrace(mesh_device.get(), tid);
}

TEST_F(UnitMeshCQTraceFixture, TensixInstantiateTraceSanity) {
    CreateDevices(2048);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Create program and add to workload
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program simple_program =
        create_simple_unary_program(*input->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
    distributed::AddProgramToMeshWorkload(workload, std::move(simple_program), device_range);

    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);
    auto tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, kNonBlocking);
    distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);

    // Instantiate a trace on a device bound command queue
    auto trace_inst = mesh_device->get_mesh_trace(tid);
    vector<uint32_t> data_fd, data_bd;

    // Backdoor read the trace buffer - using the actual device buffer
    auto device_buffer = trace_inst->mesh_buffer->get_device_buffer(distributed::MeshCoordinate{0, 0});
    detail::ReadFromBuffer(*device_buffer, data_bd);

    // Frontdoor read the trace buffer
    data_fd.resize(device_buffer->size() / sizeof(uint32_t));
    EnqueueReadBuffer(mesh_device->get_devices()[0]->command_queue(), *device_buffer, data_fd.data(), kBlocking);
    EXPECT_EQ(data_fd, data_bd);

    log_trace(LogTest, "Trace buffer content: {}", data_fd);
    distributed::ReleaseTrace(mesh_device.get(), tid);
}

TEST_F(UnitMeshCQTraceFixture, TensixEnqueueProgramTraceCapture) {
    CreateDevices(2048);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    // Create program and add to workload
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program simple_program =
        create_simple_unary_program(*input->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
    distributed::AddProgramToMeshWorkload(workload, std::move(simple_program), device_range);

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, true);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);
    distributed::ReadShard(mesh_command_queue, eager_output_data, output, distributed::MeshCoordinate{0, 0}, true);

    distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, true);

    auto tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);

    // Create and Enqueue a Program with a live trace to ensure that a warning is generated
    auto input_temp = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output_temp = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    distributed::MeshWorkload temp_workload;
    Program simple_program_temp = create_simple_unary_program(
        *input_temp->get_device_buffer(zero_coord), *output_temp->get_device_buffer(zero_coord));
    distributed::AddProgramToMeshWorkload(temp_workload, std::move(simple_program_temp), device_range);
    distributed::EnqueueMeshWorkload(mesh_command_queue, temp_workload, true);

    // Run trace that can clobber the temporary buffers created above
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), tid, true);
    distributed::ReadShard(mesh_command_queue, trace_output_data, output, distributed::MeshCoordinate{0, 0}, true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(mesh_device.get(), tid);
}

TEST_F(UnitMeshCQTraceFixture, TensixEnqueueProgramDeviceCapture) {
    CreateDevices(2048);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    bool has_eager = true;
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // EAGER MODE EXECUTION
    if (has_eager) {
        Program simple_program =
            create_simple_unary_program(*input->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
        distributed::AddProgramToMeshWorkload(workload, std::move(simple_program), device_range);

        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, true);
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);
        distributed::ReadShard(mesh_command_queue, eager_output_data, output, distributed::MeshCoordinate{0, 0}, true);
    }

    // MESH DEVICE CAPTURE AND REPLAY MODE
    bool has_trace = false;
    distributed::MeshTraceId tid;
    for (int i = 0; i < 1; i++) {
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, true);

        if (!has_trace) {
            // Program must be cached first
            tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);
            has_trace = true;
        }
        distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), tid, true);

        distributed::ReadShard(mesh_command_queue, trace_output_data, output, distributed::MeshCoordinate{0, 0}, true);
        if (has_eager) {
            EXPECT_TRUE(eager_output_data == trace_output_data);
        }
    }

    // Done
    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(mesh_device.get(), tid);
}

TEST_F(UnitMeshCQTraceFixture, TensixEnqueueTwoProgramTrace) {
    CreateDevices(6144);
    // Get command queue from device for this test, since its running in async mode
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto interm = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // Create two programs and add to workload
    distributed::MeshWorkload workload0;
    distributed::MeshWorkload workload1;
    Program op0 =
        create_simple_unary_program(*input->get_device_buffer(zero_coord), *interm->get_device_buffer(zero_coord));
    Program op1 =
        create_simple_unary_program(*interm->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));

    distributed::AddProgramToMeshWorkload(workload0, std::move(op0), device_range);
    distributed::AddProgramToMeshWorkload(workload1, std::move(op1), device_range);

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Trace mode output
    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 5);
    vector<vector<uint32_t>> trace_outputs;

    for (auto i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    // Eager mode
    vector<uint32_t> expected_output_data;
    vector<uint32_t> eager_output_data;
    expected_output_data.resize(input_data.size());
    eager_output_data.resize(input_data.size());

    // Warm up and use the eager blocking run as the expected output
    distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kBlocking);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload0, kBlocking);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload1, kBlocking);
    distributed::ReadShard(
        mesh_command_queue, expected_output_data, output, distributed::MeshCoordinate{0, 0}, kBlocking);
    distributed::Finish(mesh_command_queue);

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        for (auto i = 0; i < num_loops; i++) {
            ScopedTimer timer(mode + " loop " + std::to_string(i));
            distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, blocking);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload0, blocking);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload1, blocking);
            distributed::ReadShard(
                mesh_command_queue, eager_output_data, output, distributed::MeshCoordinate{0, 0}, blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            distributed::Finish(mesh_command_queue);
        }
        EXPECT_TRUE(eager_output_data == expected_output_data);
    }

    // Capture trace on a trace queue
    auto tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload0, kNonBlocking);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload1, kNonBlocking);
    distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kNonBlocking);
        distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), tid, kNonBlocking);
        distributed::ReadShard(
            mesh_command_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, kNonBlocking);
    }
    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(mesh_device.get(), tid);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
}

TEST_F(UnitMeshCQTraceFixture, TensixEnqueueMultiProgramTraceBenchmark) {
    CreateDevices(6144);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 4);
    uint32_t num_programs = parse_env<int>("TT_METAL_TRACE_PROGRAMS", 4);
    vector<std::shared_ptr<distributed::MeshBuffer>> interm_buffers;

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Create mesh workload with multiple programs
    vector<distributed::MeshWorkload> workloads;
    for (int i = 0; i < num_programs; i++) {
        interm_buffers.push_back(distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get()));
        distributed::MeshWorkload workload;
        Program program;
        if (i == 0) {
            program = create_simple_unary_program(
                *input->get_device_buffer(zero_coord), *interm_buffers[i]->get_device_buffer(zero_coord));
        } else if (i == (num_programs - 1)) {
            program = create_simple_unary_program(
                *interm_buffers[i - 1]->get_device_buffer(zero_coord), *output->get_device_buffer(zero_coord));
        } else {
            program = create_simple_unary_program(
                *interm_buffers[i - 1]->get_device_buffer(zero_coord),
                *interm_buffers[i]->get_device_buffer(zero_coord));
        }
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        workloads.push_back(std::move(workload));
    }

    // Eager mode
    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());

    // Trace mode output
    vector<vector<uint32_t>> trace_outputs;

    for (uint32_t i = 0; i < num_loops; i++) {
        trace_outputs.push_back({});
        trace_outputs[i].resize(input_data.size());
    }

    for (bool blocking : blocking_flags) {
        std::string mode = blocking ? "Eager-B" : "Eager-NB";
        log_info(LogTest, "Starting {} profiling with {} programs", mode, num_programs);
        for (uint32_t iter = 0; iter < num_loops; iter++) {
            ScopedTimer timer(mode + " loop " + std::to_string(iter));
            distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, blocking);
            for (uint32_t i = 0; i < num_programs; i++) {
                distributed::EnqueueMeshWorkload(mesh_command_queue, workloads[i], blocking);
            }
            distributed::ReadShard(
                mesh_command_queue, eager_output_data, output, distributed::MeshCoordinate{0, 0}, blocking);
        }
        if (not blocking) {
            // (Optional) wait for the last non-blocking command to finish
            distributed::Finish(mesh_command_queue);
        }
    }

    // Capture trace on a trace queue
    auto tid = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    for (uint32_t i = 0; i < num_programs; i++) {
        distributed::EnqueueMeshWorkload(mesh_command_queue, workloads[i], kNonBlocking);
    }
    distributed::EndTraceCapture(mesh_device.get(), mesh_command_queue.id(), tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kNonBlocking);
        distributed::ReplayTrace(mesh_device.get(), mesh_command_queue.id(), tid, kNonBlocking);
        distributed::ReadShard(
            mesh_command_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, kNonBlocking);
    }
    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(mesh_device.get(), tid);
}

}  // end namespace basic_tests

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestSimpleProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();
        this->create_kernel(program, CoreType::WORKER, true);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, ActiveEthTestSimpleProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_->get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->get_devices()[0]->id()
                     << " does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();
        this->create_kernel(program, CoreType::ETH, true);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixActiveEthTestSimpleProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_->get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->get_devices()[0]->id()
                     << " does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            this->create_kernel(program, CoreType::ETH, true);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            this->create_kernel(program, CoreType::WORKER, true);
        }

        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();
        this->create_kernel(program, CoreType::WORKER);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, ActiveEthTestProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_->get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->get_devices()[0]->id()
                     << " does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program, CoreType::ETH, false, kernel_properties);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixActiveEthTestProgramsTrace) {
    if (!does_device_have_active_eth_cores(this->device_->get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->get_devices()[0]->id()
                     << " does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
            // and the max kernel size to ensure that the kernel can fit in the ring buffer
            KernelProperties kernel_properties;
            kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
            kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::ETH, false, kernel_properties);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            KernelProperties kernel_properties;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        }
        program.set_runtime_id(i);

        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestAlternatingLargeAndSmallProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();

        KernelProperties kernel_properties;
        if (i % 2 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestLargeProgramFollowedBySmallProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();

        KernelProperties kernel_properties;
        if (i == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestLargeProgramInBetweenFiveSmallProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();

        KernelProperties kernel_properties;
        if (i % 6 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    distributed::ReleaseTrace(this->device_.get(), trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestProgramsTraceAndNoTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    std::vector<distributed::MeshTraceId> trace_ids;
    std::unordered_map<uint64_t, distributed::MeshTraceId> program_ids_to_trace_ids;

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();
        this->create_kernel(program, CoreType::WORKER);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        tt::tt_metal::distributed::MeshWorkload& workload = this->workloads[i];
        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            const distributed::MeshTraceId trace_id =
                distributed::BeginTraceCapture(this->device_.get(), mesh_command_queue.id());
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            distributed::EndTraceCapture(this->device_.get(), mesh_command_queue.id(), trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(workload.get_programs()[device_range].get_id(), trace_id);
        }
    }
    distributed::Finish(mesh_command_queue);

    for (auto& workload : this->workloads) {
        const uint64_t program_id = workload.get_programs()[device_range].get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const distributed::MeshTraceId trace_id = program_ids_to_trace_ids[program_id];
            distributed::ReplayTrace(this->device_.get(), mesh_command_queue.id(), trace_id, false);
        }
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    }

    distributed::Finish(mesh_command_queue);
    for (const distributed::MeshTraceId trace_id : trace_ids) {
        distributed::ReleaseTrace(this->device_.get(), trace_id);
    }
}

TEST_F(UnitMeshRandomProgramTraceFixture, ActiveEthTestProgramsTraceAndNoTrace) {
    if (!does_device_have_active_eth_cores(this->device_->get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->get_devices()[0]->id()
                     << " does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    std::vector<distributed::MeshTraceId> trace_ids;
    std::unordered_map<uint64_t, distributed::MeshTraceId> program_ids_to_trace_ids;

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program, CoreType::ETH, false, kernel_properties);
        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        tt::tt_metal::distributed::MeshWorkload& workload = this->workloads[i];

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            ;
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            const distributed::MeshTraceId trace_id =
                distributed::BeginTraceCapture(this->device_.get(), mesh_command_queue.id());
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            distributed::EndTraceCapture(this->device_.get(), mesh_command_queue.id(), trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(workload.get_programs()[device_range].get_id(), trace_id);
        }
    }
    distributed::Finish(mesh_command_queue);

    for (auto& workload : this->workloads) {
        const uint64_t program_id = workload.get_programs()[device_range].get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const distributed::MeshTraceId trace_id = program_ids_to_trace_ids[program_id];
            distributed::ReplayTrace(this->device_.get(), mesh_command_queue.id(), trace_id, false);
        }
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    }

    distributed::Finish(mesh_command_queue);
    for (const distributed::MeshTraceId trace_id : trace_ids) {
        distributed::ReleaseTrace(this->device_.get(), trace_id);
    }
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixActiveEthTestProgramsTraceAndNoTrace) {
    if (!does_device_have_active_eth_cores(this->device_->get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device " << this->device_->get_devices()[0]->id()
                     << " does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();
    distributed::MeshCoordinate zero_coord =
        distributed::MeshCoordinate::zero_coordinate(this->device_->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    std::vector<distributed::MeshTraceId> trace_ids;
    std::unordered_map<uint64_t, distributed::MeshTraceId> program_ids_to_trace_ids;

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram();

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
            // and the max kernel size to ensure that the kernel can fit in the ring buffer
            KernelProperties kernel_properties;
            kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
            kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::ETH, false, kernel_properties);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            KernelProperties kernel_properties;
            kernel_properties.max_num_sems = MAX_NUM_SEMS / 2;
            this->create_kernel(program, CoreType::WORKER, false, kernel_properties);
        }

        distributed::AddProgramToMeshWorkload(this->workloads[i], std::move(program), device_range);
        tt::tt_metal::distributed::MeshWorkload& workload = this->workloads[i];

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            ;
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            const distributed::MeshTraceId trace_id =
                distributed::BeginTraceCapture(this->device_.get(), mesh_command_queue.id());
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            distributed::EndTraceCapture(this->device_.get(), mesh_command_queue.id(), trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(workload.get_programs()[device_range].get_id(), trace_id);
        }
    }
    distributed::Finish(mesh_command_queue);

    for (auto& workload : this->workloads) {
        const uint64_t program_id = workload.get_programs()[device_range].get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const distributed::MeshTraceId trace_id = program_ids_to_trace_ids[program_id];
            distributed::ReplayTrace(this->device_.get(), mesh_command_queue.id(), trace_id, false);
        }
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    }

    distributed::Finish(mesh_command_queue);
    for (const distributed::MeshTraceId trace_id : trace_ids) {
        distributed::ReleaseTrace(this->device_.get(), trace_id);
    }
}

}  // namespace tt::tt_metal
