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
#include <tt-metalium/tt_metal.hpp>

#include <cstdint>
#include <vector>

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

    IDevice* dev = devices_[0]->get_devices()[0];
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
        .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
        .hw_config = experimental::DataMovementGen2Config{},
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
    tt_metal::detail::WriteToDeviceL1(dev, node, address, zeros);
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> warm_up_result(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address, sizeof(uint32_t), warm_up_result);
    ASSERT_EQ(warm_up_result[0], value);

    // Capture trace
    tt_metal::detail::WriteToDeviceL1(dev, node, address, zeros);
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), 0);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(0, trace_id);

    // Replay trace
    mesh_device->replay_mesh_trace(0, trace_id, true);
    std::vector<uint32_t> trace_result(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address, sizeof(uint32_t), trace_result);
    ASSERT_EQ(trace_result[0], value);

    mesh_device->release_mesh_trace(trace_id);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarTraceMultipleReplays) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
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
        .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
        .hw_config = experimental::DataMovementGen2Config{},
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
    tt_metal::detail::WriteToDeviceL1(dev, node, address, zeros);
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> warm_up_result(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address, sizeof(uint32_t), warm_up_result);
    ASSERT_EQ(warm_up_result[0], value);

    // Capture trace
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), 0);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(0, trace_id);

    // Replay trace
    constexpr uint32_t num_replays = 5;
    for (uint32_t i = 0; i < num_replays; i++) {
        std::vector<uint32_t> zeros(1, 0);
        tt_metal::detail::WriteToDeviceL1(dev, node, address, zeros);

        mesh_device->replay_mesh_trace(0, trace_id, true);

        std::vector<uint32_t> result(1, 0);
        tt_metal::detail::ReadFromDeviceL1(dev, node, address, sizeof(uint32_t), result);
        ASSERT_EQ(result[0], value);
    }

    mesh_device->release_mesh_trace(trace_id);
}

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, QuasarTraceMultipleReplaysAcrossCQs) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
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
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
        };
        experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
        experimental::ProgramSpec spec{
            .name = std::string("trace_across_cqs_") + kernel_id, .kernels = {dm_kernel_spec}, .work_units = {main_wu}};
        Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
        experimental::ProgramRunArgs params;
        params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = DM_KERNEL,
            .runtime_arg_values = {{node, {{"address", address}}}},
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
    tt_metal::detail::WriteToDeviceL1(dev, node, address_0, zeros);
    distributed::EnqueueMeshWorkload(cq0, wl0, true);
    std::vector<uint32_t> warm_up_0(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address_0, sizeof(uint32_t), warm_up_0);
    ASSERT_EQ(warm_up_0[0], value_0);

    distributed::MeshTraceId trace_id_0 = distributed::BeginTraceCapture(mesh_device.get(), 0);
    distributed::EnqueueMeshWorkload(cq0, wl0, false);
    mesh_device->end_mesh_trace(0, trace_id_0);

    // Warm up + capture the CQ1 trace.
    tt_metal::detail::WriteToDeviceL1(dev, node, address_1, zeros);
    distributed::EnqueueMeshWorkload(cq1, wl1, true);
    std::vector<uint32_t> warm_up_1(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address_1, sizeof(uint32_t), warm_up_1);
    ASSERT_EQ(warm_up_1[0], value_1);

    distributed::MeshTraceId trace_id_1 = distributed::BeginTraceCapture(mesh_device.get(), 1);
    distributed::EnqueueMeshWorkload(cq1, wl1, false);
    mesh_device->end_mesh_trace(1, trace_id_1);

    // Interleave replays of both CQs' traces and verify each lands its own value each round.
    constexpr uint32_t num_replays = 5;
    for (uint32_t i = 0; i < num_replays; i++) {
        tt_metal::detail::WriteToDeviceL1(dev, node, address_0, zeros);
        tt_metal::detail::WriteToDeviceL1(dev, node, address_1, zeros);

        mesh_device->replay_mesh_trace(0, trace_id_0, true);
        mesh_device->replay_mesh_trace(1, trace_id_1, true);

        std::vector<uint32_t> result_0(1, 0);
        tt_metal::detail::ReadFromDeviceL1(dev, node, address_0, sizeof(uint32_t), result_0);
        ASSERT_EQ(result_0[0], value_0);

        std::vector<uint32_t> result_1(1, 0);
        tt_metal::detail::ReadFromDeviceL1(dev, node, address_1, sizeof(uint32_t), result_1);
        ASSERT_EQ(result_1[0], value_1);
    }

    mesh_device->release_mesh_trace(trace_id_0);
    mesh_device->release_mesh_trace(trace_id_1);
}
