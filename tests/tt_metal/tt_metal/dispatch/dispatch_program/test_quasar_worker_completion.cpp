// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "context/metal_context.hpp"
#include "dispatch/dispatch_core_manager.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/sub_device.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tt::tt_metal {
namespace {

constexpr uint32_t local_l1_size = 3200;

std::vector<experimental::NodeCoord> worker_nodes(const CoreCoord& worker_grid) {
    return experimental::grid_to_nodes(
        experimental::NodeCoord{0, 0}, experimental::NodeCoord{worker_grid.x - 1, worker_grid.y - 1});
}

SubDevice sub_device_from_nodes(const std::vector<experimental::NodeCoord>& nodes) {
    CoreRangeSet cores;
    for (const experimental::NodeCoord& node : nodes) {
        cores = cores.merge(CoreRangeSet(CoreRange(node, node)));
    }
    return SubDevice(std::array{cores});
}

distributed::MeshWorkload create_l1_write_workload(
    distributed::MeshDevice& mesh_device,
    const experimental::NodeCoord& node,
    uint32_t address,
    uint32_t value,
    const std::string& kernel_id) {
    const experimental::KernelSpecName kernel_name{"worker_completion_" + kernel_id};
    experimental::ProgramSpec program_spec{
        .name = "worker_completion_" + kernel_id,
        .kernels = {experimental::KernelSpec{
            .unique_id = kernel_name,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config = experimental::DataMovementGen2Config{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "worker_completion_write", .kernels = {kernel_name}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, program_spec);
    experimental::SetProgramRunArgs(
        program,
        experimental::ProgramRunArgs{
            .kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = kernel_name,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"address", address}}),
                .common_runtime_arg_values = {{"value", value}}}}});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    return workload;
}

distributed::MeshWorkload create_blank_workload(
    distributed::MeshDevice& mesh_device, const experimental::NodeCoord& node) {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreCoord{node.x, node.y},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    return workload;
}

uint32_t read_l1_word(distributed::MeshDevice& mesh_device, const experimental::NodeCoord& node, uint32_t address) {
    std::vector<uint32_t> output(1, 0);
    slow_dispatch::ReadFromL1(mesh_device, node, address, sizeof(uint32_t), output);
    return output[0];
}

void clear_l1_word(distributed::MeshDevice& mesh_device, const experimental::NodeCoord& node, uint32_t address) {
    std::vector<uint32_t> zero(1, 0);
    slow_dispatch::WriteToL1(mesh_device, node, address, zero);
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, WorkerGoTransportSingleCQ) {
    MetalContext& metal_context = MetalContext::instance();
    if (!metal_context.rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "Requires the Quasar simulator or emulator";
    }
    if (!metal_context.rtoptions().get_fast_dispatch()) {
        GTEST_SKIP() << "Requires fast dispatch";
    }
    if (metal_context.get_dispatch_core_manager().get_dispatch_core_type() != CoreType::DISPATCH) {
        GTEST_SKIP() << "Requires dispatch engines";
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const std::vector<experimental::NodeCoord> nodes = worker_nodes(worker_grid);
    if (nodes.size() < 2) {
        GTEST_SKIP() << "Requires at least two worker nodes";
    }

    const SubDeviceManagerId subset_manager =
        mesh_device->create_sub_device_manager({sub_device_from_nodes({nodes[0]})}, local_l1_size);
    const size_t partition = nodes.size() / 2;
    const std::vector<experimental::NodeCoord> first_partition(nodes.begin(), nodes.begin() + partition);
    const std::vector<experimental::NodeCoord> second_partition(nodes.begin() + partition, nodes.end());
    const SubDeviceManagerId split_manager = mesh_device->create_sub_device_manager(
        {sub_device_from_nodes(first_partition), sub_device_from_nodes(second_partition)}, local_l1_size);

    const uint32_t l1_address =
        metal_context.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    distributed::MeshCommandQueue& command_queue = mesh_device->mesh_command_queue();

    distributed::MeshWorkload default_blank = create_blank_workload(*mesh_device, nodes[0]);
    distributed::MeshWorkload default_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x11110000, "single_default");
    distributed::EnqueueMeshWorkload(command_queue, default_blank, false);
    distributed::EnqueueMeshWorkload(command_queue, default_write, false);

    constexpr uint32_t num_back_to_back_writes = 4;
    for (uint32_t write_index = 0; write_index < num_back_to_back_writes; ++write_index) {
        const uint32_t address = l1_address + sizeof(uint32_t) * (write_index + 1);
        const uint32_t value = 0x22220000 + write_index;
        distributed::MeshWorkload workload = create_l1_write_workload(
            *mesh_device, nodes[0], address, value, "single_back_to_back_" + std::to_string(write_index));
        distributed::EnqueueMeshWorkload(command_queue, workload, false);
    }
    distributed::Finish(command_queue);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x11110000u);
    for (uint32_t write_index = 0; write_index < num_back_to_back_writes; ++write_index) {
        const uint32_t address = l1_address + sizeof(uint32_t) * (write_index + 1);
        EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], address), 0x22220000u + write_index);
    }

    // A one-device mesh cannot produce an idle-device go for a program assigned only to another device.

    distributed::MeshWorkload traced_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x33330000, "single_trace");
    distributed::EnqueueMeshWorkload(command_queue, traced_write, true);
    clear_l1_word(*mesh_device, nodes[0], l1_address);
    const distributed::MeshTraceId trace_id = mesh_device->begin_mesh_trace(command_queue);
    distributed::EnqueueMeshWorkload(command_queue, traced_write, false);
    mesh_device->end_mesh_trace(command_queue, trace_id);
    mesh_device->replay_mesh_trace(command_queue, trace_id, true);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x33330000u);
    mesh_device->release_mesh_trace(trace_id);

    mesh_device->load_sub_device_manager(subset_manager);
    distributed::MeshWorkload subset_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x44440000, "single_subset");
    distributed::EnqueueMeshWorkload(command_queue, subset_write, false);
    distributed::Finish(command_queue);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x44440000u);

    mesh_device->load_sub_device_manager(split_manager);
    distributed::MeshWorkload first_split_write =
        create_l1_write_workload(*mesh_device, first_partition[0], l1_address, 0x55550000, "single_split_first");
    distributed::MeshWorkload second_split_write =
        create_l1_write_workload(*mesh_device, second_partition[0], l1_address, 0x55550001, "single_split_second");
    distributed::EnqueueMeshWorkload(command_queue, first_split_write, false);
    distributed::EnqueueMeshWorkload(command_queue, second_split_write, false);
    distributed::Finish(command_queue);
    EXPECT_EQ(read_l1_word(*mesh_device, first_partition[0], l1_address), 0x55550000u);
    EXPECT_EQ(read_l1_word(*mesh_device, second_partition[0], l1_address), 0x55550001u);

    mesh_device->clear_loaded_sub_device_manager();
    distributed::MeshWorkload restored_default_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x66660000, "single_restored_default");
    distributed::EnqueueMeshWorkload(command_queue, restored_default_write, false);
    distributed::Finish(command_queue);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x66660000u);
}

TEST_F(QuasarMultiCQMeshDeviceSingleCardFixture, WorkerCompletionTransport) {
    MetalContext& metal_context = MetalContext::instance();
    if (!metal_context.rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "Requires the Quasar simulator or emulator";
    }
    if (!metal_context.rtoptions().get_fast_dispatch()) {
        GTEST_SKIP() << "Requires fast dispatch";
    }
    if (metal_context.get_dispatch_core_manager().get_dispatch_core_type() != CoreType::DISPATCH) {
        GTEST_SKIP() << "Requires dispatch engines";
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const std::vector<experimental::NodeCoord> nodes = worker_nodes(worker_grid);
    if (nodes.size() < 2) {
        GTEST_SKIP() << "Requires at least two worker nodes";
    }

    const SubDeviceManagerId combined_manager =
        mesh_device->create_sub_device_manager({sub_device_from_nodes(nodes)}, local_l1_size);
    std::vector<SubDevice> split_sub_devices;
    split_sub_devices.reserve(nodes.size());
    for (const experimental::NodeCoord& node : nodes) {
        split_sub_devices.push_back(sub_device_from_nodes({node}));
    }
    const SubDeviceManagerId split_manager = mesh_device->create_sub_device_manager(split_sub_devices, local_l1_size);

    const uint32_t l1_address =
        metal_context.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    distributed::MeshCommandQueue& command_queue_0 = mesh_device->mesh_command_queue(0);
    distributed::MeshCommandQueue& command_queue_1 = mesh_device->mesh_command_queue(1);

    mesh_device->load_sub_device_manager(combined_manager);
    clear_l1_word(*mesh_device, nodes[0], l1_address);
    distributed::MeshWorkload combined_blank = create_blank_workload(*mesh_device, nodes[0]);
    distributed::MeshWorkload combined_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x11110001, "combined");
    distributed::EnqueueMeshWorkload(command_queue_0, combined_blank, false);
    distributed::EnqueueMeshWorkload(command_queue_0, combined_write, false);
    distributed::MeshEvent combined_event = command_queue_0.enqueue_record_event_to_host();
    distributed::EventSynchronize(combined_event);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x11110001u);

    mesh_device->load_sub_device_manager(split_manager);
    distributed::MeshWorkload group_one_blank = create_blank_workload(*mesh_device, nodes[0]);
    distributed::MeshWorkload group_one_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x22220001, "group_one");
    distributed::MeshWorkload group_two_blank = create_blank_workload(*mesh_device, nodes[1]);
    distributed::MeshWorkload group_two_write =
        create_l1_write_workload(*mesh_device, nodes[1], l1_address, 0x22220002, "group_two");
    distributed::EnqueueMeshWorkload(command_queue_0, group_one_blank, false);
    distributed::EnqueueMeshWorkload(command_queue_0, group_one_write, false);
    distributed::EnqueueMeshWorkload(command_queue_0, group_two_blank, false);
    distributed::EnqueueMeshWorkload(command_queue_0, group_two_write, false);
    distributed::Finish(command_queue_0);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x22220001u);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[1], l1_address), 0x22220002u);

    distributed::MeshWorkload traced_write =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x33330001, "trace");
    distributed::EnqueueMeshWorkload(command_queue_0, traced_write, true);
    clear_l1_word(*mesh_device, nodes[0], l1_address);
    const distributed::MeshTraceId trace_id = mesh_device->begin_mesh_trace(command_queue_0);
    distributed::EnqueueMeshWorkload(command_queue_0, traced_write, false);
    mesh_device->end_mesh_trace(command_queue_0, trace_id);
    mesh_device->replay_mesh_trace(command_queue_0, trace_id, true);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x33330001u);
    mesh_device->release_mesh_trace(trace_id);

    distributed::MeshWorkload producer =
        create_l1_write_workload(*mesh_device, nodes[0], l1_address, 0x44440001, "producer");
    distributed::MeshWorkload consumer =
        create_l1_write_workload(*mesh_device, nodes[1], l1_address, 0x44440002, "consumer");
    distributed::EnqueueMeshWorkload(command_queue_0, producer, false);
    distributed::MeshEvent handoff_event = command_queue_0.enqueue_record_event();
    command_queue_1.enqueue_wait_for_event(handoff_event);
    distributed::EnqueueMeshWorkload(command_queue_1, consumer, false);
    distributed::Finish(command_queue_0);
    distributed::Finish(command_queue_1);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[0], l1_address), 0x44440001u);
    EXPECT_EQ(read_l1_word(*mesh_device, nodes[1], l1_address), 0x44440002u);
}

}  // namespace tt::tt_metal
