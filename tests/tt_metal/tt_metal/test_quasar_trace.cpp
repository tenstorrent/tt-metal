// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"
#include "experimental/metal2_host_api/data_movement_hardware_config.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <cstdint>
#include <vector>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarTraceSingleReplay) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t value = 0xcafe1234;

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 2,
        .compile_time_args = {{"cached_write", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "trace_test", .kernels = {dm_kernel_spec}, .work_units = {main_wu}};

    distributed::MeshWorkload workload;
    workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));
    Program& prog = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
        .common_runtime_arg_values = {{"value", value}},
    }};
    experimental::SetProgramRunArgs(prog, params);

    // Warm up
    std::vector<uint32_t> zeros(1, 0);
    slow_dispatch::WriteToL1(*mesh_device, node, address, zeros);
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> warm_up_result(1, 0);
    slow_dispatch::ReadFromL1(*mesh_device, node, address, sizeof(uint32_t), warm_up_result);
    ASSERT_EQ(warm_up_result[0], value);

    // Capture trace
    slow_dispatch::WriteToL1(*mesh_device, node, address, zeros);
    distributed::MeshTraceId trace_id = mesh_device->begin_mesh_trace(cq);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(cq, trace_id);

    // Replay trace
    mesh_device->replay_mesh_trace(cq, trace_id, true);
    std::vector<uint32_t> trace_result(1, 0);
    slow_dispatch::ReadFromL1(*mesh_device, node, address, sizeof(uint32_t), trace_result);
    ASSERT_EQ(trace_result[0], value);

    mesh_device->release_mesh_trace(trace_id);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarTraceMultipleReplays) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t value = 0x5a5a5a5a;

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 2,
        .compile_time_args = {{"cached_write", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{
        .name = "trace_multi_replay_test", .kernels = {dm_kernel_spec}, .work_units = {main_wu}};

    distributed::MeshWorkload workload;
    workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));
    Program& prog = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
        .common_runtime_arg_values = {{"value", value}},
    }};
    experimental::SetProgramRunArgs(prog, params);

    // Warm up
    std::vector<uint32_t> zeros(1, 0);
    slow_dispatch::WriteToL1(*mesh_device, node, address, zeros);
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> warm_up_result(1, 0);
    slow_dispatch::ReadFromL1(*mesh_device, node, address, sizeof(uint32_t), warm_up_result);
    ASSERT_EQ(warm_up_result[0], value);

    // Capture trace
    distributed::MeshTraceId trace_id = mesh_device->begin_mesh_trace(cq);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(cq, trace_id);

    // Replay trace
    constexpr uint32_t num_replays = 5;
    for (uint32_t i = 0; i < num_replays; i++) {
        std::vector<uint32_t> zeros(1, 0);
        slow_dispatch::WriteToL1(*mesh_device, node, address, zeros);

        mesh_device->replay_mesh_trace(cq, trace_id, true);

        std::vector<uint32_t> result(1, 0);
        slow_dispatch::ReadFromL1(*mesh_device, node, address, sizeof(uint32_t), result);
        ASSERT_EQ(result[0], value);
    }

    mesh_device->release_mesh_trace(trace_id);
}

// Each trace must replay the DFB config that was current at its own capture. Change the DFB's
// entry size between two captures and check each replay reports its own entry size.
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarTraceDFBSizeOverride) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};
    constexpr uint32_t capture_a_entry_size = 1024;
    constexpr uint32_t capture_b_entry_size = 2048;
    constexpr uint32_t num_entries = 16;

    // dfb_extent_probe_* write an 8-word extent record per snapshot; word 0 is the entry size.
    constexpr uint32_t extent_record_bytes = 8 * sizeof(uint32_t);
    const uint32_t l1_alignment = mesh_device->allocator()->get_alignment(BufferType::L1);
    const uint32_t records_bytes = tt::align(2 * extent_record_bytes, l1_alignment);
    const uint32_t producer_record_address = static_cast<uint32_t>(mesh_device->l1_size_per_core()) - records_bytes;
    const uint32_t consumer_record_address = producer_record_address + extent_record_bytes;

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::DFBSpecName DFB{"dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = capture_a_entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_extent_probe_dm.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .accessor_name = "out",
              .endpoint_type = experimental::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .compile_time_args = {{"num_tc_snapshots", 1u}, {"rotate_tc", 0u}, {"credits_to_post", 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_2xx =
                    experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                        .disable_dfb_implicit_sync_for = {DFB},
                    },
            },
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/dfb_extent_probe_compute.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = DFB,
              .accessor_name = "in",
              .endpoint_type = experimental::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .compile_time_args =
            {{"num_tc_snapshots", 1u},
             {"rotate_tc", 0u},
             {"drain_producer_rotate_credits", 0u},
             {"drain_last_tc_credit", 0u},
             {"num_producers", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};
    experimental::ProgramSpec spec{
        .name = "trace_dfb_size_override",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .work_units = {main_wu},
    };

    distributed::MeshWorkload workload;
    workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));
    Program& prog = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", producer_record_address}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                experimental::MakeRuntimeArgsForSingleNode(node, {{"result_l1_addr", consumer_record_address}}),
        },
    };
    experimental::SetProgramRunArgs(prog, params);

    std::vector<uint32_t> zeros(2 * extent_record_bytes / sizeof(uint32_t), 0);
    auto run_and_expect_entry_size = [&](auto run, uint32_t expected_entry_size) {
        slow_dispatch::WriteToL1(*mesh_device, node, producer_record_address, zeros);
        run();
        std::vector<uint32_t> records;
        slow_dispatch::ReadFromL1(*mesh_device, node, producer_record_address, 2 * extent_record_bytes, records);
        EXPECT_EQ(records[0], expected_entry_size) << "producer";
        EXPECT_EQ(records[extent_record_bytes / sizeof(uint32_t)], expected_entry_size) << "consumer";
    };
    auto capture_trace = [&] {
        const distributed::MeshTraceId trace_id = mesh_device->begin_mesh_trace(cq);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        mesh_device->end_mesh_trace(cq, trace_id);
        return trace_id;
    };

    // Warm up
    run_and_expect_entry_size([&] { distributed::EnqueueMeshWorkload(cq, workload, true); }, capture_a_entry_size);

    const distributed::MeshTraceId trace_id_a = capture_trace();

    // No untraced run after the override: only trace capture re-serializes the DFB config.
    params.dfb_run_overrides.push_back({.dfb = DFB, .entry_size = capture_b_entry_size});
    experimental::SetProgramRunArgs(prog, params);
    const distributed::MeshTraceId trace_id_b = capture_trace();

    run_and_expect_entry_size([&] { mesh_device->replay_mesh_trace(cq, trace_id_a, true); }, capture_a_entry_size);
    run_and_expect_entry_size([&] { mesh_device->replay_mesh_trace(cq, trace_id_b, true); }, capture_b_entry_size);

    mesh_device->release_mesh_trace(trace_id_a);
    mesh_device->release_mesh_trace(trace_id_b);
}

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, QuasarTraceMultipleReplaysAcrossCQs) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};

    const uint32_t address_0 = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    // Separate the two CQs' outputs by a cache line so neither DM cache flush touches the other.
    const uint32_t address_1 = address_0 + 64;
    const uint32_t value_0 = 0xcafe0000;
    const uint32_t value_1 = 0xcafe1111;

    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    auto make_workload = [&](uint32_t address, uint32_t value, const char* kernel_id) {
        distributed::MeshWorkload wl;
        const experimental::KernelSpecName DM_KERNEL{kernel_id};
        experimental::KernelSpec dm_kernel_spec{
            .unique_id = DM_KERNEL,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
            .num_threads = 2,
            .compile_time_args = {{"cached_write", 1u}},
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        };
        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
        experimental::ProgramSpec spec{
            .name = std::string("trace_across_cqs_") + kernel_id, .kernels = {dm_kernel_spec}, .work_units = {main_wu}};
        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_KERNEL,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
            .common_runtime_arg_values = {{"value", value}},
        }};
        experimental::SetProgramRunArgs(program, params);
        wl.add_program(device_range, std::move(program));
        return wl;
    };

    auto wl0 = make_workload(address_0, value_0, "trace_dm_0");
    auto wl1 = make_workload(address_1, value_1, "trace_dm_1");

    distributed::MeshCommandQueue& cq0 = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& cq1 = mesh_device->mesh_command_queue(1);

    std::vector<uint32_t> zeros(1, 0);

    // Warm up + capture the CQ0 trace.
    slow_dispatch::WriteToL1(*mesh_device, node, address_0, zeros);
    distributed::EnqueueMeshWorkload(cq0, wl0, true);
    std::vector<uint32_t> warm_up_0(1, 0);
    slow_dispatch::ReadFromL1(*mesh_device, node, address_0, sizeof(uint32_t), warm_up_0);
    ASSERT_EQ(warm_up_0[0], value_0);

    distributed::MeshTraceId trace_id_0 = mesh_device->begin_mesh_trace(cq0);
    distributed::EnqueueMeshWorkload(cq0, wl0, false);
    mesh_device->end_mesh_trace(cq0, trace_id_0);

    // Warm up + capture the CQ1 trace.
    slow_dispatch::WriteToL1(*mesh_device, node, address_1, zeros);
    distributed::EnqueueMeshWorkload(cq1, wl1, true);
    std::vector<uint32_t> warm_up_1(1, 0);
    slow_dispatch::ReadFromL1(*mesh_device, node, address_1, sizeof(uint32_t), warm_up_1);
    ASSERT_EQ(warm_up_1[0], value_1);

    distributed::MeshTraceId trace_id_1 = mesh_device->begin_mesh_trace(cq1);
    distributed::EnqueueMeshWorkload(cq1, wl1, false);
    mesh_device->end_mesh_trace(cq1, trace_id_1);

    // Interleave replays of both CQs' traces and verify each lands its own value each round.
    constexpr uint32_t num_replays = 5;
    for (uint32_t i = 0; i < num_replays; i++) {
        slow_dispatch::WriteToL1(*mesh_device, node, address_0, zeros);
        slow_dispatch::WriteToL1(*mesh_device, node, address_1, zeros);

        mesh_device->replay_mesh_trace(cq0, trace_id_0, true);
        mesh_device->replay_mesh_trace(cq1, trace_id_1, true);

        std::vector<uint32_t> result_0(1, 0);
        slow_dispatch::ReadFromL1(*mesh_device, node, address_0, sizeof(uint32_t), result_0);
        ASSERT_EQ(result_0[0], value_0);

        std::vector<uint32_t> result_1(1, 0);
        slow_dispatch::ReadFromL1(*mesh_device, node, address_1, sizeof(uint32_t), result_1);
        ASSERT_EQ(result_1[0], value_1);
    }

    mesh_device->release_mesh_trace(trace_id_0);
    mesh_device->release_mesh_trace(trace_id_1);
}
