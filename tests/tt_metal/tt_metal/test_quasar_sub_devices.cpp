// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"
#include "hostdevcommon/common_values.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/sub_device.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kLocalL1Size = 3200;

std::vector<experimental::NodeCoord> worker_nodes(const CoreCoord& worker_grid) {
    return experimental::grid_to_nodes(
        experimental::NodeCoord{0, 0}, experimental::NodeCoord{worker_grid.x - 1, worker_grid.y - 1});
}

std::vector<std::vector<experimental::NodeCoord>> partition_worker_nodes(
    const std::vector<experimental::NodeCoord>& nodes) {
    const size_t num_partitions = std::min(nodes.size(), static_cast<size_t>(DISPATCH_MAX_MESSAGE_ENTRIES));
    std::vector<std::vector<experimental::NodeCoord>> partitions;
    partitions.reserve(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
        const auto begin = nodes.begin() + nodes.size() * i / num_partitions;
        const auto end = nodes.begin() + nodes.size() * (i + 1) / num_partitions;
        partitions.emplace_back(begin, end);
    }
    return partitions;
}

CoreRangeSet node_set(const std::vector<experimental::NodeCoord>& nodes) {
    CoreRangeSet cores;
    for (const auto& node : nodes) {
        cores = cores.merge(CoreRangeSet(CoreRange(node, node)));
    }
    return cores;
}

SubDevice single_node_sub_device(const CoreCoord& node) {
    return SubDevice(std::array{CoreRangeSet(CoreRange(node, node))});
}

SubDevice sub_device_from_nodes(const std::vector<experimental::NodeCoord>& nodes) {
    return SubDevice(std::array{node_set(nodes)});
}

SubDevice full_grid_sub_device(const CoreCoord& worker_grid) {
    return SubDevice(std::array{CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}))});
}

template <typename TargetNodes>
distributed::MeshWorkload create_l1_write_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const TargetNodes& target_nodes,
    const std::vector<std::pair<experimental::NodeCoord, uint32_t>>& node_addresses,
    uint32_t value,
    const std::string& id) {
    const experimental::KernelSpecName kernel_name{"l1_writer_" + id};
    experimental::ProgramSpec spec{
        .name = "sub_device_l1_write_" + id,
        .kernels = {experimental::KernelSpec{
            .unique_id = kernel_name,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}},
            .hw_config = experimental::DataMovementGen2Config{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "writer_" + id, .kernels = {kernel_name}, .target_nodes = target_nodes}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::ProgramRunArgs::KernelRunArgs kernel_args{
        .kernel = kernel_name, .common_runtime_arg_values = {{"value", value}}};
    for (const auto& [node, address] : node_addresses) {
        experimental::AddRuntimeArgsForNode(kernel_args.runtime_arg_values, node, {{"address", address}});
    }
    experimental::SetProgramRunArgs(program, experimental::ProgramRunArgs{.kernel_run_args = {kernel_args}});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    return workload;
}

uint32_t read_l1_word(IDevice* device, const experimental::NodeCoord& node, uint32_t address) {
    std::vector<uint32_t> output(1, 0);
    detail::ReadFromDeviceL1(device, node, address, sizeof(uint32_t), output);
    return output[0];
}

struct SyncWorkloads {
    distributed::MeshWorkload waiter;
    distributed::MeshWorkload syncer;
    distributed::MeshWorkload incrementer;
    GlobalSemaphore semaphore;
};

SyncWorkloads create_sync_workloads(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const SubDevice& incrementer_sub_device,
    const SubDevice& waiter_sub_device) {
    const auto waiter_node = waiter_sub_device.cores(HalProgrammableCoreType::TENSIX).ranges().front().start_coord;
    const auto& incrementer_nodes = incrementer_sub_device.cores(HalProgrammableCoreType::TENSIX);
    const auto syncer_node = incrementer_nodes.ranges().back().end_coord;
    const auto waiter_physical = mesh_device->worker_core_from_logical_core(waiter_node);
    const auto syncer_physical = mesh_device->worker_core_from_logical_core(syncer_node);
    const auto all_nodes = CoreRangeSet(CoreRange(waiter_node, waiter_node))
                               .merge(incrementer_nodes)
                               .merge(CoreRangeSet(CoreRange(syncer_node, syncer_node)));
    auto semaphore = CreateGlobalSemaphore(mesh_device.get(), all_nodes, 0);

    const experimental::KernelSpecName waiter_kernel{"quasar_sub_device_waiter"};
    experimental::ProgramSpec waiter_spec{
        .name = "quasar_sub_device_waiter",
        .kernels = {experimental::KernelSpec{
            .unique_id = waiter_kernel,
            .source = "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/persistent_waiter.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"sem_addr", "num_inc", "sync_core_x", "sync_core_y"}},
            .hw_config = experimental::DataMovementGen2Config{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "waiter", .kernels = {waiter_kernel}, .target_nodes = experimental::NodeCoord(waiter_node)}},
    };
    Program waiter_program = experimental::MakeProgramFromSpec(*mesh_device, waiter_spec);
    experimental::SetProgramRunArgs(
        waiter_program,
        experimental::ProgramRunArgs{
            .kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = waiter_kernel,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    experimental::NodeCoord(waiter_node),
                    {{"sem_addr", semaphore.address()},
                     {"num_inc", incrementer_nodes.num_cores()},
                     {"sync_core_x", syncer_physical.x},
                     {"sync_core_y", syncer_physical.y}})}}});

    const experimental::KernelSpecName syncer_kernel{"quasar_sub_device_syncer"};
    experimental::ProgramSpec syncer_spec{
        .name = "quasar_sub_device_syncer",
        .kernels = {experimental::KernelSpec{
            .unique_id = syncer_kernel,
            .source = "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"sem_addr"}},
            .hw_config = experimental::DataMovementGen2Config{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "syncer", .kernels = {syncer_kernel}, .target_nodes = experimental::NodeCoord(syncer_node)}},
    };
    Program syncer_program = experimental::MakeProgramFromSpec(*mesh_device, syncer_spec);
    experimental::SetProgramRunArgs(
        syncer_program,
        experimental::ProgramRunArgs{
            .kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = syncer_kernel,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    experimental::NodeCoord(syncer_node), {{"sem_addr", semaphore.address()}})}}});

    const experimental::KernelSpecName incrementer_kernel{"quasar_sub_device_incrementer"};
    experimental::ProgramSpec incrementer_spec{
        .name = "quasar_sub_device_incrementer",
        .kernels = {experimental::KernelSpec{
            .unique_id = incrementer_kernel,
            .source = "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/incrementer.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"sem_addr", "waiter_core_x", "waiter_core_y"}},
            .hw_config = experimental::DataMovementGen2Config{}}},
        .work_units = {experimental::WorkUnitSpec{
            .name = "incrementer", .kernels = {incrementer_kernel}, .target_nodes = incrementer_nodes}},
    };
    Program incrementer_program = experimental::MakeProgramFromSpec(*mesh_device, incrementer_spec);
    experimental::ProgramRunArgs::KernelRunArgs incrementer_args{.kernel = incrementer_kernel};
    for (const auto& node : experimental::node_range_to_nodes(incrementer_nodes)) {
        experimental::AddRuntimeArgsForNode(
            incrementer_args.runtime_arg_values,
            node,
            {{"sem_addr", semaphore.address()},
             {"waiter_core_x", waiter_physical.x},
             {"waiter_core_y", waiter_physical.y}});
    }
    experimental::SetProgramRunArgs(
        incrementer_program, experimental::ProgramRunArgs{.kernel_run_args = {std::move(incrementer_args)}});

    const distributed::MeshCoordinateRange device_range(mesh_device->shape());
    SyncWorkloads result{.semaphore = std::move(semaphore)};
    result.waiter.add_program(device_range, std::move(waiter_program));
    result.syncer.add_program(device_range, std::move(syncer_program));
    result.incrementer.add_program(device_range, std::move(incrementer_program));
    return result;
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, TestSubDevicePartitionsWorkerGrid) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    const auto nodes = worker_nodes(worker_grid);
    const auto partitions = partition_worker_nodes(nodes);

    std::vector<SubDevice> sub_devices;
    sub_devices.reserve(partitions.size());
    for (const auto& partition : partitions) {
        sub_devices.push_back(sub_device_from_nodes(partition));
    }
    const auto manager = mesh_device->create_sub_device_manager(sub_devices, kLocalL1Size);
    mesh_device->load_sub_device_manager(manager);

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<distributed::MeshWorkload> workloads;
    workloads.reserve(partitions.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
        const uint32_t value = 0x11110000u + (static_cast<uint32_t>(i) << 16);
        std::vector<std::pair<experimental::NodeCoord, uint32_t>> node_addresses;
        node_addresses.reserve(partitions[i].size());
        for (const auto& node : partitions[i]) {
            node_addresses.emplace_back(node, address);
        }
        workloads.push_back(create_l1_write_workload(
            mesh_device,
            experimental::NodeRangeSet{node_set(partitions[i])},
            node_addresses,
            value,
            "partition_" + std::to_string(i)));
        distributed::EnqueueMeshWorkload(cq, workloads.back(), /*blocking=*/i + 1 == partitions.size());
    }

    IDevice* device = mesh_device->get_devices()[0];
    for (size_t i = 0; i < partitions.size(); ++i) {
        const uint32_t value = 0x11110000u + (static_cast<uint32_t>(i) << 16);
        for (const auto& node : partitions[i]) {
            EXPECT_EQ(read_l1_word(device, node, address), value);
        }
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TestSubDeviceShardedL1BufferAllocation) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    constexpr uint32_t page_size = 64;

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    const auto nodes = worker_nodes(worker_grid);
    const auto partitions = partition_worker_nodes(nodes);

    std::vector<SubDevice> sub_devices;
    sub_devices.reserve(partitions.size());
    for (const auto& partition : partitions) {
        sub_devices.push_back(sub_device_from_nodes(partition));
    }
    const auto manager = mesh_device->create_sub_device_manager(sub_devices, kLocalL1Size);
    mesh_device->load_sub_device_manager(manager);

    auto make_buffer = [&](const std::vector<experimental::NodeCoord>& partition, SubDeviceId sub_device_id) {
        const CoreRangeSet grid = node_set(partition);
        ShardSpecBuffer shard_spec(
            grid, {1, 16}, ShardOrientation::ROW_MAJOR, {1, 16}, {static_cast<uint32_t>(partition.size()), 1});
        distributed::DeviceLocalBufferConfig local_config{
            .page_size = page_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(std::move(shard_spec), TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = false,
            .sub_device_id = sub_device_id};
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = partition.size() * page_size}, local_config, mesh_device.get());
    };

    const DeviceAddr l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const DeviceAddr local_l1_limit = l1_unreserved_base + kLocalL1Size;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> buffers;
    buffers.reserve(partitions.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
        buffers.push_back(make_buffer(partitions[i], SubDeviceId{static_cast<uint8_t>(i)}));
        EXPECT_GE(buffers.back()->address(), l1_unreserved_base);
        EXPECT_LE(
            buffers.back()->address() + buffers.back()->get_backing_buffer()->aligned_page_size(), local_l1_limit);
        ASSERT_EQ(buffers.back()->device_local_config().sub_device_id, SubDeviceId{static_cast<uint8_t>(i)});
        EXPECT_EQ(buffers.back()->device_local_config().sharding_args.shard_spec()->grid(), node_set(partitions[i]));
    }
    for (size_t i = 0; i < partitions.size(); ++i) {
        for (size_t j = i + 1; j < partitions.size(); ++j) {
            EXPECT_NE(
                buffers[i]->device_local_config().sharding_args.shard_spec()->grid(),
                buffers[j]->device_local_config().sharding_args.shard_spec()->grid());
        }
    }

    auto& cq = mesh_device->mesh_command_queue();
    std::vector<std::vector<uint32_t>> inputs(partitions.size());
    std::vector<std::vector<uint32_t>> outputs(partitions.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
        inputs[i].assign(
            partitions[i].size() * page_size / sizeof(uint32_t), 0x11110000u + (static_cast<uint32_t>(i) << 16));
        distributed::EnqueueWriteMeshBuffer(cq, buffers[i], inputs[i]);
    }
    for (size_t i = 0; i < partitions.size(); ++i) {
        distributed::EnqueueReadMeshBuffer(cq, outputs[i], buffers[i], /*blocking=*/true);
        EXPECT_EQ(outputs[i], inputs[i]);
    }

    IDevice* device = mesh_device->get_devices()[0];
    for (size_t i = 0; i < partitions.size(); ++i) {
        for (const auto& node : partitions[i]) {
            EXPECT_EQ(read_l1_word(device, node, buffers[i]->address()), inputs[i].front());
        }
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TestSubDeviceStallGroup) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    const auto nodes = worker_nodes(worker_grid);
    // Waiter on the last node; all other worker nodes form the incrementer sub-device.
    const experimental::NodeCoord waiter_node = nodes.back();
    std::vector<experimental::NodeCoord> incrementer_node_list(nodes.begin(), nodes.end() - 1);
    const auto incrementer_sub_device = sub_device_from_nodes(incrementer_node_list);
    const auto waiter_sub_device = single_node_sub_device(waiter_node);
    const auto manager =
        mesh_device->create_sub_device_manager({incrementer_sub_device, waiter_sub_device}, kLocalL1Size);
    mesh_device->load_sub_device_manager(manager);
    auto sync = create_sync_workloads(mesh_device, incrementer_sub_device, waiter_sub_device);

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, sync.waiter, /*blocking=*/false);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});
    distributed::EnqueueMeshWorkload(cq, sync.syncer, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(cq, sync.incrementer, /*blocking=*/false);
    mesh_device->reset_sub_device_stall_group();
    distributed::Finish(cq);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TestSubDeviceManagerSwitching) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_nodes = worker_grid.x * worker_grid.y;
    if (num_nodes < 2) {
        GTEST_SKIP() << "Test requires at least two worker nodes";
    }

    const auto nodes = worker_nodes(worker_grid);
    const auto partitions = partition_worker_nodes(nodes);

    std::vector<SubDevice> split_sub_devices;
    split_sub_devices.reserve(partitions.size());
    for (const auto& partition : partitions) {
        split_sub_devices.push_back(sub_device_from_nodes(partition));
    }
    const auto combined_manager =
        mesh_device->create_sub_device_manager({full_grid_sub_device(worker_grid)}, kLocalL1Size);
    const auto split_manager = mesh_device->create_sub_device_manager(split_sub_devices, kLocalL1Size);
    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    IDevice* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();

    mesh_device->load_sub_device_manager(combined_manager);
    std::vector<std::pair<experimental::NodeCoord, uint32_t>> combined_addresses;
    combined_addresses.reserve(nodes.size());
    for (const auto& node : nodes) {
        combined_addresses.emplace_back(node, address);
    }
    auto combined_workload = create_l1_write_workload(
        mesh_device,
        experimental::NodeRange{{0, 0}, {worker_grid.x - 1, worker_grid.y - 1}},
        combined_addresses,
        0x33330000,
        "combined_manager");
    distributed::EnqueueMeshWorkload(cq, combined_workload, /*blocking=*/true);
    for (const auto& node : nodes) {
        EXPECT_EQ(read_l1_word(device, node, address), 0x33330000u);
    }

    mesh_device->load_sub_device_manager(split_manager);
    std::vector<distributed::MeshWorkload> split_workloads;
    split_workloads.reserve(partitions.size());
    for (size_t i = 0; i < partitions.size(); ++i) {
        const uint32_t value = 0x44440000u + (static_cast<uint32_t>(i) << 16);
        std::vector<std::pair<experimental::NodeCoord, uint32_t>> node_addresses;
        node_addresses.reserve(partitions[i].size());
        for (const auto& node : partitions[i]) {
            node_addresses.emplace_back(node, address);
        }
        split_workloads.push_back(create_l1_write_workload(
            mesh_device,
            experimental::NodeRangeSet{node_set(partitions[i])},
            node_addresses,
            value,
            "split_manager_" + std::to_string(i)));
        distributed::EnqueueMeshWorkload(cq, split_workloads.back(), /*blocking=*/i + 1 == partitions.size());
    }
    for (size_t i = 0; i < partitions.size(); ++i) {
        const uint32_t value = 0x44440000u + (static_cast<uint32_t>(i) << 16);
        for (const auto& node : partitions[i]) {
            EXPECT_EQ(read_l1_word(device, node, address), value);
        }
    }
}
