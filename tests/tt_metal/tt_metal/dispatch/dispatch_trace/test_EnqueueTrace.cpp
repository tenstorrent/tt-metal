// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/fmt.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/compute_hardware_config.hpp>
#include <tt-metalium/workload/specification/data_movement_hardware_config.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include "dispatch_test_utils.hpp"
#include "env_lib.hpp"
#include "gtest/gtest.h"
#include "hostdevcommon/kernel_structs.h"
#include "trace/trace_buffer.hpp"
#include "multi_command_queue_fixture.hpp"
#include "random_program_fixture.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/common/scoped_timer.hpp"
#include <tt-umd/types/core_coordinates.hpp>
#include "distributed/mesh_trace.hpp"

// Access to internal API: ProgramImpl::get_id
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;

Program create_simple_unary_program(distributed::MeshDevice& mesh_device, Buffer& input, Buffer& output) {
    const NodeCoord node{0, 0};
    const DFBSpecName input_dfb{"input_dfb"};
    const DFBSpecName output_dfb{"output_dfb"};
    const KernelSpecName reader{"reader"};
    const KernelSpecName writer{"writer"};
    const KernelSpecName compute{"compute"};
    const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t num_tiles = input.num_pages();
    const DataMovementHardwareConfig::DataMovement2XXConfig explicit_sync_config{
        .disable_dfb_implicit_sync_for_all = true,
    };

    const KernelSpec reader_spec{
        .unique_id = reader,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = input_dfb,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
            .access_pattern = DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "bank_id", "num_tiles"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
                .config_2xx = explicit_sync_config,
            },
    };
    const KernelSpec writer_spec{
        .unique_id = writer,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = output_dfb,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::CONSUMER,
            .access_pattern = DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_1xx =
                    DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
                .config_2xx = explicit_sync_config,
            },
    };
    const KernelSpec compute_spec{
        .unique_id = compute,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options =
            {.defines = {{"SFPU_OP_EXP_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
        .dfb_bindings =
            {{
                 .dfb_spec_name = input_dfb,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .access_pattern = DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = output_dfb,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .access_pattern = DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"per_core_block_cnt", num_tiles}, {"per_core_block_size", 1u}},
        .hw_config = ComputeHardwareConfig{.sfpu_precision_mode = Precision::Approximate},
    };
    const ProgramSpec spec{
        .name = "simple_unary",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {DataflowBufferSpec{
                 .unique_id = input_dfb,
                 .entry_size = tile_size,
                 .num_entries = num_tiles,
                 .data_format_metadata = tt::DataFormat::Float16_b,
             },
             DataflowBufferSpec{
                 .unique_id = output_dfb,
                 .entry_size = tile_size,
                 .num_entries = num_tiles,
                 .data_format_metadata = tt::DataFormat::Float16_b,
             }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {reader, writer, compute}, .target_nodes = node}},
    };
    Program program = Program(mesh_device, spec);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = reader,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", static_cast<uint32_t>(input.address())}, {"bank_id", 0u}, {"num_tiles", num_tiles}}),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = writer,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<uint32_t>(output.address())},
                 {"bank_id", 0u},
                 {"num_tiles", output.num_pages()}}),
        },
    };
    program.set_program_run_args(run_args);
    return program;
}

// All basic trace tests just assert that the replayed result exactly matches
// the eager mode results
namespace basic_tests {

constexpr bool kBlocking = true;
constexpr bool kNonBlocking = false;
vector<bool> blocking_flags = {kBlocking, kNonBlocking};

TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueOneProgramTrace) {
    CreateDevice(2048);

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, this->device_.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, this->device_.get());

    auto& mesh_command_queue = this->device_->mesh_command_queue(0);
    auto& data_movement_queue = this->device_->mesh_command_queue(1);

    // Create program and add to workload
    distributed::MeshWorkload workload;
    Program simple_program = create_simple_unary_program(
        *this->device_, *input->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));
    workload.add_program(device_range_, std::move(simple_program));

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
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

    auto tid = this->device_->begin_mesh_trace(mesh_command_queue);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    this->device_->end_mesh_trace(mesh_command_queue, tid);

    this->device_->replay_mesh_trace(mesh_command_queue, tid, true);
    distributed::ReadShard(data_movement_queue, trace_output_data, output, distributed::MeshCoordinate{0, 0}, true);
    // distributed::EnqueueReadMeshBuffer(data_movement_queue, trace_output_data, output, true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(tid);  //?
}

TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueOneProgramTraceLoops) {
    CreateDevice(4096);

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, this->device_.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, this->device_.get());

    auto& mesh_command_queue = this->device_->mesh_command_queue(0);
    auto& data_movement_queue = this->device_->mesh_command_queue(1);

    // Create program and add to workload
    distributed::MeshWorkload workload;
    Program simple_program = create_simple_unary_program(
        *this->device_, *input->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));
    workload.add_program(device_range_, std::move(simple_program));

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
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
            trace_id = this->device_->begin_mesh_trace(mesh_command_queue);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            this->device_->end_mesh_trace(mesh_command_queue, trace_id);
            trace_captured = true;
        }

        this->device_->replay_mesh_trace(mesh_command_queue, trace_id, false);
        distributed::ReadShard(data_movement_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, true);

        // Expect same output across all loops
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }

    // Done
    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueOneProgramTraceBenchmark) {
    CreateDevice(6144);

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, this->device_.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, this->device_.get());

    constexpr bool kBlocking = true;
    constexpr bool kNonBlocking = false;
    vector<bool> blocking_flags = {kBlocking, kNonBlocking};

    // Single Q for data and commands
    // Keep this queue in passthrough mode for now
    auto& mesh_command_queue = this->device_->mesh_command_queue(0);

    // Create program and add to workload
    distributed::MeshWorkload workload;
    Program simple_program = create_simple_unary_program(
        *this->device_, *input->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));
    workload.add_program(device_range_, std::move(simple_program));

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
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
    auto tid = this->device_->begin_mesh_trace(mesh_command_queue);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    this->device_->end_mesh_trace(mesh_command_queue, tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        tt::ScopedTimer timer("Trace loop " + std::to_string(i));
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kNonBlocking);
        this->device_->replay_mesh_trace(mesh_command_queue, tid, kNonBlocking);
        distributed::ReadShard(
            mesh_command_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, kNonBlocking);
    }
    distributed::Finish(mesh_command_queue);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
    this->device_->release_mesh_trace(tid);
}

TEST_F(UnitMeshCQTraceFixture, TensixInstantiateTraceSanity) {
    CreateDevices(2048);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Create program and add to workload
    distributed::MeshWorkload workload;
    Program simple_program = create_simple_unary_program(
        *mesh_device, *input->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));
    workload.add_program(device_range_, std::move(simple_program));

    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);
    auto tid = mesh_device->begin_mesh_trace(mesh_command_queue);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, kNonBlocking);
    mesh_device->end_mesh_trace(mesh_command_queue, tid);

    // Instantiate a trace on a device bound command queue
    auto trace_inst = mesh_device->get_mesh_trace(tid);
    vector<uint32_t> data_fd, data_bd;

    // Backdoor read the trace buffer
    slow_dispatch::ReadFromBuffer(*trace_inst->mesh_buffer, data_bd);

    // Frontdoor read the trace buffer
    data_fd.resize(data_bd.size());
    distributed::ReadShard(
        mesh_command_queue, data_fd, trace_inst->mesh_buffer, distributed::MeshCoordinate{0, 0}, kBlocking);
    EXPECT_EQ(data_fd, data_bd);

    log_trace(LogTest, "Trace buffer content: {}", data_fd);
    mesh_device->release_mesh_trace(tid);
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
    Program simple_program = create_simple_unary_program(
        *mesh_device, *input->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));
    workload.add_program(device_range_, std::move(simple_program));

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
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

    auto tid = mesh_device->begin_mesh_trace(mesh_command_queue);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    mesh_device->end_mesh_trace(mesh_command_queue, tid);

    // Create and Enqueue a Program with a live trace to ensure that a warning is generated
    auto input_temp = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output_temp = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    distributed::MeshWorkload temp_workload;
    Program simple_program_temp = create_simple_unary_program(
        *mesh_device, *input_temp->get_device_buffer(zero_coord_), *output_temp->get_device_buffer(zero_coord_));
    temp_workload.add_program(device_range_, std::move(simple_program_temp));
    distributed::EnqueueMeshWorkload(mesh_command_queue, temp_workload, true);

    // Run trace that can clobber the temporary buffers created above
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    mesh_device->replay_mesh_trace(mesh_command_queue, tid, true);
    distributed::ReadShard(mesh_command_queue, trace_output_data, output, distributed::MeshCoordinate{0, 0}, true);
    EXPECT_TRUE(eager_output_data == trace_output_data);

    // Done
    distributed::Finish(mesh_command_queue);
    mesh_device->release_mesh_trace(tid);
}

TEST_F(UnitMeshCQTraceFixture, TensixEnqueueProgramDeviceCapture) {
    CreateDevices(2048);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    vector<uint32_t> eager_output_data;
    eager_output_data.resize(input_data.size());
    vector<uint32_t> trace_output_data;
    trace_output_data.resize(input_data.size());

    bool has_eager = true;
    distributed::MeshWorkload workload;

    // EAGER MODE EXECUTION
    if (has_eager) {
        Program simple_program = create_simple_unary_program(
            *mesh_device, *input->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));
        workload.add_program(device_range_, std::move(simple_program));

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
            tid = mesh_device->begin_mesh_trace(mesh_command_queue);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            mesh_device->end_mesh_trace(mesh_command_queue, tid);
            has_trace = true;
        }
        mesh_device->replay_mesh_trace(mesh_command_queue, tid, true);

        distributed::ReadShard(mesh_command_queue, trace_output_data, output, distributed::MeshCoordinate{0, 0}, true);
        if (has_eager) {
            EXPECT_TRUE(eager_output_data == trace_output_data);
        }
    }

    // Done
    distributed::Finish(mesh_command_queue);
    mesh_device->release_mesh_trace(tid);
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

    // Create two programs and add to workload
    distributed::MeshWorkload workload0;
    distributed::MeshWorkload workload1;
    Program op0 = create_simple_unary_program(
        *mesh_device, *input->get_device_buffer(zero_coord_), *interm->get_device_buffer(zero_coord_));
    Program op1 = create_simple_unary_program(
        *mesh_device, *interm->get_device_buffer(zero_coord_), *output->get_device_buffer(zero_coord_));

    workload0.add_program(device_range_, std::move(op0));
    workload1.add_program(device_range_, std::move(op1));

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
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
    auto tid = mesh_device->begin_mesh_trace(mesh_command_queue);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload0, kNonBlocking);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload1, kNonBlocking);
    mesh_device->end_mesh_trace(mesh_command_queue, tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kNonBlocking);
        mesh_device->replay_mesh_trace(mesh_command_queue, tid, kNonBlocking);
        distributed::ReadShard(
            mesh_command_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, kNonBlocking);
    }
    distributed::Finish(mesh_command_queue);
    mesh_device->release_mesh_trace(tid);

    // Expect same output across all loops
    for (auto i = 0; i < num_loops; i++) {
        EXPECT_TRUE(trace_outputs[i] == trace_outputs[0]);
    }
}

TEST_F(UnitMeshCQTraceFixture, TensixEnqueueMultiProgramTraceBenchmark) {
    uint32_t num_loops = parse_env<int>("TT_METAL_TRACE_LOOPS", 4);
    uint32_t num_programs = parse_env<int>("TT_METAL_TRACE_PROGRAMS", 4);
    constexpr uint32_t trace_bytes_per_program = 2048;
    CreateDevices(num_programs * trace_bytes_per_program + kMaxTraceBufPageSize);
    auto mesh_device = this->devices_[0];
    auto& mesh_command_queue = mesh_device->mesh_command_queue();

    distributed::ReplicatedBufferConfig replicated_config{.size = 2048};
    distributed::DeviceLocalBufferConfig device_config{.page_size = 2048, .buffer_type = BufferType::DRAM};
    auto input = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());
    auto output = distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get());

    vector<std::shared_ptr<distributed::MeshBuffer>> interm_buffers;

    vector<uint32_t> input_data(input->get_device_buffer(zero_coord_)->size() / sizeof(uint32_t), 0);
    for (uint32_t i = 0; i < input_data.size(); i++) {
        input_data[i] = i;
    }

    // Create mesh workload with multiple programs
    vector<distributed::MeshWorkload> workloads;
    for (int i = 0; i < num_programs; i++) {
        interm_buffers.push_back(distributed::MeshBuffer::create(replicated_config, device_config, mesh_device.get()));
        distributed::MeshWorkload workload;
        Program program = CreateProgram(*mesh_device);
        if (i == 0) {
            program = create_simple_unary_program(
                *mesh_device,
                *input->get_device_buffer(zero_coord_),
                *interm_buffers[i]->get_device_buffer(zero_coord_));
        } else if (i == (num_programs - 1)) {
            program = create_simple_unary_program(
                *mesh_device,
                *interm_buffers[i - 1]->get_device_buffer(zero_coord_),
                *output->get_device_buffer(zero_coord_));
        } else {
            program = create_simple_unary_program(
                *mesh_device,
                *interm_buffers[i - 1]->get_device_buffer(zero_coord_),
                *interm_buffers[i]->get_device_buffer(zero_coord_));
        }
        workload.add_program(device_range_, std::move(program));
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
    auto tid = mesh_device->begin_mesh_trace(mesh_command_queue);
    for (uint32_t i = 0; i < num_programs; i++) {
        distributed::EnqueueMeshWorkload(mesh_command_queue, workloads[i], kNonBlocking);
    }
    mesh_device->end_mesh_trace(mesh_command_queue, tid);

    // Trace mode execution
    for (auto i = 0; i < num_loops; i++) {
        ScopedTimer timer("Trace loop " + std::to_string(i));
        distributed::EnqueueWriteMeshBuffer(mesh_command_queue, input, input_data, kNonBlocking);
        mesh_device->replay_mesh_trace(mesh_command_queue, tid, kNonBlocking);
        distributed::ReadShard(
            mesh_command_queue, trace_outputs[i], output, distributed::MeshCoordinate{0, 0}, kNonBlocking);
    }
    distributed::Finish(mesh_command_queue);
    mesh_device->release_mesh_trace(tid);
}

}  // end namespace basic_tests

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestSimpleProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->workloads[i].add_program(device_range_, this->create_program_with_kernel(CoreType::WORKER, true));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, ActiveEthTestSimpleProgramsTrace) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (!does_device_have_active_eth_cores(this->device_->impl().get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram(*this->device_);
        this->create_kernel(program, CoreType::ETH, true);
        this->workloads[i].add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixActiveEthTestSimpleProgramsTrace) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (!does_device_have_active_eth_cores(this->device_->impl().get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram(*this->device_);

        bool eth_kernel_added_to_program = false;
        if (rand() % 2 == 0) {
            this->create_kernel(program, CoreType::ETH, true);
            eth_kernel_added_to_program = true;
        }
        if (rand() % 2 == 0 || !eth_kernel_added_to_program) {
            this->create_kernel(program, CoreType::WORKER, true);
        }

        this->workloads[i].add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, NIGHTLY_TensixTestProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->workloads[i].add_program(device_range_, this->create_program_with_kernel(CoreType::WORKER));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, ActiveEthTestProgramsTrace) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (!does_device_have_active_eth_cores(this->device_->impl().get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram(*this->device_);
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program, CoreType::ETH, false, kernel_properties);
        this->workloads[i].add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixActiveEthTestProgramsTrace) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (!does_device_have_active_eth_cores(this->device_->impl().get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram(*this->device_);

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

        this->workloads[i].add_program(device_range_, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, NIGHTLY_TensixTestAlternatingLargeAndSmallProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        KernelProperties kernel_properties;
        if (i % 2 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->workloads[i].add_program(
            device_range_, this->create_program_with_kernel(CoreType::WORKER, false, kernel_properties));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, NIGHTLY_TensixTestLargeProgramFollowedBySmallProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        KernelProperties kernel_properties;
        if (i == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->workloads[i].add_program(
            device_range_, this->create_program_with_kernel(CoreType::WORKER, false, kernel_properties));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestLargeProgramInBetweenFiveSmallProgramsTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        KernelProperties kernel_properties;
        if (i % 6 == 0) {
            kernel_properties = this->get_large_kernel_properties();
        } else {
            kernel_properties = this->get_small_kernel_properties();
        }

        this->workloads[i].add_program(
            device_range_, this->create_program_with_kernel(CoreType::WORKER, false, kernel_properties));
        distributed::EnqueueMeshWorkload(mesh_command_queue, this->workloads[i], false);
    }

    const distributed::MeshTraceId trace_id = this->trace_programs();

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixTestProgramsTraceAndNoTrace) {
    auto& mesh_command_queue = this->device_->mesh_command_queue();

    std::vector<distributed::MeshTraceId> trace_ids;
    std::unordered_map<uint64_t, distributed::MeshTraceId> program_ids_to_trace_ids;

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        this->workloads[i].add_program(device_range_, this->create_program_with_kernel(CoreType::WORKER));
        tt::tt_metal::distributed::MeshWorkload& workload = this->workloads[i];
        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            const distributed::MeshTraceId trace_id = this->device_->begin_mesh_trace(mesh_command_queue);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            this->device_->end_mesh_trace(mesh_command_queue, trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(workload.get_programs()[device_range_].impl().get_id(), trace_id);
        }
    }
    distributed::Finish(mesh_command_queue);

    for (auto& workload : this->workloads) {
        const uint64_t program_id = workload.get_programs()[device_range_].impl().get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const distributed::MeshTraceId trace_id = program_ids_to_trace_ids[program_id];
            this->device_->replay_mesh_trace(mesh_command_queue, trace_id, false);
        }
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    }

    distributed::Finish(mesh_command_queue);
    for (const distributed::MeshTraceId trace_id : trace_ids) {
        this->device_->release_mesh_trace(trace_id);
    }
}

TEST_F(UnitMeshRandomProgramTraceFixture, ActiveEthTestProgramsTraceAndNoTrace) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (!does_device_have_active_eth_cores(this->device_->impl().get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();

    std::vector<distributed::MeshTraceId> trace_ids;
    std::unordered_map<uint64_t, distributed::MeshTraceId> program_ids_to_trace_ids;

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram(*this->device_);
        // Large eth kernels currently don't fit in the ring buffer, so we're reducing the max number of RTAs
        // and the max kernel size to ensure that the kernel can fit in the ring buffer
        KernelProperties kernel_properties;
        kernel_properties.max_kernel_size_bytes = MAX_KERNEL_SIZE_BYTES / 2;
        kernel_properties.max_num_rt_args = MAX_NUM_RUNTIME_ARGS / 4;
        this->create_kernel(program, CoreType::ETH, false, kernel_properties);
        this->workloads[i].add_program(device_range_, std::move(program));
        tt::tt_metal::distributed::MeshWorkload& workload = this->workloads[i];

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            ;
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            const distributed::MeshTraceId trace_id = this->device_->begin_mesh_trace(mesh_command_queue);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            this->device_->end_mesh_trace(mesh_command_queue, trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(workload.get_programs()[device_range_].impl().get_id(), trace_id);
        }
    }
    distributed::Finish(mesh_command_queue);

    for (auto& workload : this->workloads) {
        const uint64_t program_id = workload.get_programs()[device_range_].impl().get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const distributed::MeshTraceId trace_id = program_ids_to_trace_ids[program_id];
            this->device_->replay_mesh_trace(mesh_command_queue, trace_id, false);
        }
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    }

    distributed::Finish(mesh_command_queue);
    for (const distributed::MeshTraceId trace_id : trace_ids) {
        this->device_->release_mesh_trace(trace_id);
    }
}

TEST_F(UnitMeshRandomProgramTraceFixture, TensixActiveEthTestProgramsTraceAndNoTrace) {
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    if (!does_device_have_active_eth_cores(this->device_->impl().get_devices()[0])) {
        GTEST_SKIP() << "Skipping test because device does not have any active ethernet cores";
    }

    auto& mesh_command_queue = this->device_->mesh_command_queue();

    std::vector<distributed::MeshTraceId> trace_ids;
    std::unordered_map<uint64_t, distributed::MeshTraceId> program_ids_to_trace_ids;

    for (uint32_t i = 0; i < NUM_WORKLOADS; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Creating Program {}", i);
        }
        Program program = CreateProgram(*this->device_);

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

        this->workloads[i].add_program(device_range_, std::move(program));
        tt::tt_metal::distributed::MeshWorkload& workload = this->workloads[i];

        const bool use_trace = (rand() % 2) == 0;
        if (use_trace) {
            ;
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            const distributed::MeshTraceId trace_id = this->device_->begin_mesh_trace(mesh_command_queue);
            distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
            this->device_->end_mesh_trace(mesh_command_queue, trace_id);
            trace_ids.push_back(trace_id);
            program_ids_to_trace_ids.emplace(workload.get_programs()[device_range_].impl().get_id(), trace_id);
        }
    }
    distributed::Finish(mesh_command_queue);

    for (auto& workload : this->workloads) {
        const uint64_t program_id = workload.get_programs()[device_range_].impl().get_id();
        const bool use_trace = program_ids_to_trace_ids.contains(program_id);
        if (use_trace) {
            const distributed::MeshTraceId trace_id = program_ids_to_trace_ids[program_id];
            this->device_->replay_mesh_trace(mesh_command_queue, trace_id, false);
        }
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    }

    distributed::Finish(mesh_command_queue);
    for (const distributed::MeshTraceId trace_id : trace_ids) {
        this->device_->release_mesh_trace(trace_id);
    }
}

// Enqueues the same DFB program twice inside a single trace with different DFB
// configs between the two uses.  After replay, reads the config values each
// execution wrote to L1 and verifies they match the configs that were active at
// capture time.
TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixEnqueueDFBProgramTrace) {
    constexpr uint32_t entry_size_a = 1024;
    constexpr uint32_t num_entries_a = 16;
    constexpr uint32_t entry_size_b = 512;
    constexpr uint32_t num_entries_b = 32;
    static_assert(entry_size_a * num_entries_a == entry_size_b * num_entries_b, "total DFB size must be equal");
    constexpr uint32_t dfb_total_size = entry_size_a * num_entries_a;

    CreateDevice(dfb_total_size * 4);

    const NodeCoord node{0, 0};
    constexpr uint32_t extent_record_bytes = 8 * sizeof(uint32_t);
    constexpr uint32_t extent_record_words = extent_record_bytes / sizeof(uint32_t);
    const uint32_t l1_alignment = this->device_->allocator()->get_alignment(BufferType::L1);
    const uint32_t records_bytes = tt::align(4 * extent_record_bytes, l1_alignment);
    const uint32_t producer_record_address_a = static_cast<uint32_t>(this->device_->l1_size_per_core()) - records_bytes;
    const uint32_t consumer_record_address_a = producer_record_address_a + extent_record_bytes;
    const uint32_t producer_record_address_b = consumer_record_address_a + extent_record_bytes;
    const uint32_t consumer_record_address_b = producer_record_address_b + extent_record_bytes;

    const DFBSpecName dfb{"dfb"};
    const KernelSpecName producer{"producer"};
    const KernelSpecName consumer{"consumer"};
    const ProgramSpec spec{
        .name = "dfb_run_overrides_in_trace",
        .kernels =
            {KernelSpec{
                 .unique_id = producer,
                 .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_extent_probe_dm.cpp",
                 .num_threads = 1,
                 .dfb_bindings =
                     {{.dfb_spec_name = dfb,
                       .accessor_name = "out",
                       .endpoint_type = DFBEndpointType::PRODUCER,
                       .access_pattern = DFBAccessPattern::STRIDED}},
                 .compile_time_args = {{"num_tc_snapshots", 1u}, {"rotate_tc", 0u}, {"credits_to_post", 0u}},
                 .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
                 .hw_config =
                     DataMovementHardwareConfig{
                         .config_1xx =
                             DataMovementHardwareConfig::DataMovement1XXConfig{
                                 .processor = DataMovementProcessor::RISCV_0,
                                 .noc = NOC::RISCV_0_default,
                             },
                         .config_2xx =
                             DataMovementHardwareConfig::DataMovement2XXConfig{
                                 .disable_dfb_implicit_sync_for = {dfb},
                             },
                     },
             },
             KernelSpec{
                 .unique_id = consumer,
                 .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_extent_probe_compute.cpp",
                 .num_threads = 1,
                 .dfb_bindings =
                     {{.dfb_spec_name = dfb,
                       .accessor_name = "in",
                       .endpoint_type = DFBEndpointType::CONSUMER,
                       .access_pattern = DFBAccessPattern::STRIDED}},
                 .compile_time_args =
                     {{"num_tc_snapshots", 1u},
                      {"rotate_tc", 0u},
                      {"drain_producer_rotate_credits", 0u},
                      {"drain_last_tc_credit", 0u},
                      {"num_producers", 1u}},
                 .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
                 .hw_config = ComputeHardwareConfig{},
             }},
        .dataflow_buffers = {DataflowBufferSpec{
            .unique_id = dfb,
            .entry_size = entry_size_a,
            .num_entries = num_entries_a,
            .data_format_metadata = tt::DataFormat::Float16_b,
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {producer, consumer}, .target_nodes = node}},
    };

    distributed::MeshWorkload workload;
    workload.add_program(device_range_, Program(*this->device_, spec));
    Program& program = workload.get_programs().at(device_range_);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = producer,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", producer_record_address_a}}),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = consumer,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", consumer_record_address_a}}),
        },
    };
    program.set_program_run_args(run_args);

    distributed::MeshCommandQueue& mesh_command_queue = this->device_->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, true);
    {
        vector<uint32_t> records;
        slow_dispatch::ReadFromL1(*this->device_, node, producer_record_address_a, 2 * extent_record_bytes, records);
        EXPECT_EQ(records[0], entry_size_a) << "eager: producer entry_size mismatch";
        EXPECT_EQ(records[extent_record_words], entry_size_a) << "eager: consumer entry_size mismatch";
    }

    const distributed::MeshTraceId trace_id = this->device_->begin_mesh_trace(mesh_command_queue);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = producer,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", producer_record_address_b}}),
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = consumer,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", consumer_record_address_b}}),
        },
    };
    run_args.dfb_run_overrides = {{.dfb = dfb, .entry_size = entry_size_b, .num_entries = num_entries_b}};
    program.set_program_run_args(run_args);
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, false);
    this->device_->end_mesh_trace(mesh_command_queue, trace_id);

    vector<uint32_t> zeros(records_bytes / sizeof(uint32_t), 0);
    slow_dispatch::WriteToL1(*this->device_, node, producer_record_address_a, zeros);

    this->device_->replay_mesh_trace(mesh_command_queue, trace_id, true);

    vector<uint32_t> records;
    slow_dispatch::ReadFromL1(*this->device_, node, producer_record_address_a, 4 * extent_record_bytes, records);
    EXPECT_EQ(records[0], entry_size_a) << "trace execution 1: producer entry_size mismatch";
    EXPECT_EQ(records[extent_record_words], entry_size_a) << "trace execution 1: consumer entry_size mismatch";
    EXPECT_EQ(records[2 * extent_record_words], entry_size_b) << "trace execution 2: producer entry_size mismatch";
    EXPECT_EQ(records[3 * extent_record_words], entry_size_b) << "trace execution 2: consumer entry_size mismatch";

    distributed::Finish(mesh_command_queue);
    this->device_->release_mesh_trace(trace_id);
}

}  // namespace tt::tt_metal
