// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <thread>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "dispatch_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include "llrt.hpp"
#include "multi_command_queue_fixture.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include "sub_device_test_utils.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/distributed.hpp>

// Access to internal API: ProgramImpl::validate_circular_buffer_region
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

constexpr uint32_t k_local_l1_size = 3200;
const std::string k_coordinates_kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/read_my_coordinates.cpp";

static tt_metal::Program create_blank_program(const CoreRange& cores) {
    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreRangeSet(cores),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    return program;
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceCBAllocation) {
    auto mesh_device = devices_[0];
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});
    SubDevice sub_device_1(std::array{sharded_cores_1});
    auto sub_device_manager_1 = mesh_device->create_sub_device_manager({sub_device_1}, k_local_l1_size);
    DeviceAddr l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    DeviceAddr l1_max_size = mesh_device->l1_size_per_core();
    DeviceAddr l1_total_size = l1_max_size - l1_unreserved_base;
    mesh_device->load_sub_device_manager(sub_device_manager_1);
    // Program-local CBs are DRAM-aligned from persistent high-water. Leave three
    // local-L1 slots so a 1x CB fits and a 4x CB overlaps the top-down global buffer.
    uint32_t global_buffer_size = l1_total_size - (k_local_l1_size * 3);
    ShardSpecBuffer global_shard_spec_buffer =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});

    distributed::ReplicatedBufferConfig replicated_config_1 = {
        sharded_cores_1.num_cores() * global_buffer_size,
    };
    distributed::DeviceLocalBufferConfig local_config_1 = {
        .page_size = global_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(global_shard_spec_buffer, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto global_buffer = distributed::MeshBuffer::create(replicated_config_1, local_config_1, mesh_device.get());
    Program program = CreateProgram();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(k_local_l1_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, hal::get_l1_alignment());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, sharded_cores_1, cb_src0_config);

    program.impl().prepare_ephemeral_allocation(mesh_device.get());
    program.impl().allocate_circular_buffers(mesh_device.get());
    program.impl().validate_circular_buffer_region(mesh_device.get());
    UpdateCircularBufferTotalSize(program, cb_src0, k_local_l1_size * 4);
    program.impl().prepare_ephemeral_allocation(mesh_device.get());
    program.impl().allocate_circular_buffers(mesh_device.get());
    EXPECT_THROW(program.impl().validate_circular_buffer_region(mesh_device.get()), std::exception);
    global_buffer.reset();
    program.impl().validate_circular_buffer_region(mesh_device.get());
    program.impl().compile_and_allocate(mesh_device.get(), false);
    program.impl().compile_and_allocate(mesh_device.get(), false);  // cached layout, still no L1 collision
    ShardSpecBuffer local_shard_spec_buffer =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});
    distributed::DeviceLocalBufferConfig local_config_2 = {
        .page_size = global_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(local_shard_spec_buffer, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };

    auto local_buffer = distributed::MeshBuffer::create(replicated_config_1, local_config_2, mesh_device.get());
    EXPECT_THROW(program.impl().validate_circular_buffer_region(mesh_device.get()), std::exception);
    EXPECT_THROW(program.impl().compile_and_allocate(mesh_device.get(), false), std::exception);
    UpdateCircularBufferTotalSize(program, cb_src0, k_local_l1_size / 4);
    program.impl().prepare_ephemeral_allocation(mesh_device.get());
    program.impl().allocate_circular_buffers(mesh_device.get());
    program.impl().validate_circular_buffer_region(mesh_device.get());
    EXPECT_NO_THROW(program.impl().compile_and_allocate(mesh_device.get(), false));
    EXPECT_NO_THROW(program.impl().compile_and_allocate(mesh_device.get(), false));
    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager_1);
}

void test_sub_device_synchronization(distributed::MeshDevice* device) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);
    uint32_t page_size_1 = 32;
    distributed::ReplicatedBufferConfig replicated_config_1 = {
        sharded_cores_1.num_cores() * page_size_1,
    };
    ShardSpecBuffer shard_spec_buffer_1 =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});

    distributed::DeviceLocalBufferConfig local_config_1 = {
        .page_size = page_size_1,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec_buffer_1, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto input_1 = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, sharded_cores_1.num_cores() * page_size_1 / sizeof(uint32_t));

    std::array sub_device_ids_to_block = {SubDeviceId{0}};

    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);

    std::vector<CoreCoord> physical_cores_1;
    physical_cores_1.reserve(sharded_cores_1_vec.size());
    for (const auto& core : sharded_cores_1_vec) {
        physical_cores_1.push_back(device->worker_core_from_logical_core(core));
    }

    device->load_sub_device_manager(sub_device_manager);

    auto [program, syncer_core, global_semaphore] = create_single_sync_program(device, sub_device_2);
    distributed::MeshWorkload mesh_workload;
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    mesh_workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
    device->set_sub_device_stall_group(sub_device_ids_to_block);

    auto buffer_1 = distributed::MeshBuffer::create(replicated_config_1, local_config_1, device);

    // Test blocking synchronize doesn't stall
    distributed::Synchronize(*device, std::nullopt);

    // Test blocking write buffer doesn't stall
    distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), buffer_1, input_1, true);

    // Test record event won't cause a stall

    auto event = device->mesh_command_queue().enqueue_record_event_to_host();
    distributed::Synchronize(*device, std::nullopt);

    // Test blocking read buffer doesn't stall
    std::vector<uint32_t> output_1;
    distributed::ReadShard(device->mesh_command_queue(), output_1, buffer_1, zero_coord, true);
    EXPECT_EQ(input_1, output_1);
    const auto device_id = device->get_device_ids()[0];
    auto input_1_it = input_1.begin();
    for (const auto& physical_core : physical_cores_1) {
        auto readback = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            device_id, physical_core, buffer_1->address(), page_size_1);
        EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
        input_1_it += page_size_1 / sizeof(uint32_t);
    }
    auto sem_addr = global_semaphore.address();
    auto physical_syncer_core = device->worker_core_from_logical_core(syncer_core);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device_id, physical_syncer_core, std::vector<uint32_t>{1}, sem_addr);

    // Full synchronization
    device->reset_sub_device_stall_group();
    distributed::Finish(device->mesh_command_queue());
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceSynchronization) {
    test_sub_device_synchronization(devices_[0].get());
}

TEST_F(UnitMeshMultiCQSingleDeviceFixture, TensixTestMultiCQSubDeviceSynchronization) {
    test_sub_device_synchronization(device_.get());
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceBasicPrograms) {
    constexpr uint32_t k_num_iters = 5;
    auto mesh_device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager);

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < k_num_iters; i++) {
        // Create fresh programs for each iteration
        auto [waiter_program, syncer_program, incrementer_program, global_sem] =
            create_basic_sync_program(mesh_device.get(), sub_device_1, sub_device_2);

        distributed::MeshWorkload waiter_mesh_workload;
        waiter_mesh_workload.add_program(device_range, std::move(waiter_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_mesh_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});

        // Test blocking on one sub-device
        distributed::MeshWorkload syncer_mesh_workload;
        syncer_mesh_workload.add_program(device_range, std::move(syncer_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_mesh_workload, true);

        distributed::MeshWorkload incrementer_mesh_workload;
        incrementer_mesh_workload.add_program(device_range, std::move(incrementer_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_mesh_workload, false);

        mesh_device->reset_sub_device_stall_group();
    }
    distributed::Synchronize(*mesh_device, std::nullopt);
    ReadMeshDeviceProfilerResults(*mesh_device);
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceBasicProgramsReuse) {
    constexpr uint32_t k_num_iters = 5;
    auto mesh_device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    // sub-device 3 and 4 are supersets of sub-device 1 and 2 respectively
    SubDevice sub_device_3(std::array{CoreRangeSet(std::vector{CoreRange({0, 0}, {2, 2}), CoreRange({5, 5}, {5, 5})})});
    SubDevice sub_device_4(std::array{
        CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4}), CoreRange({6, 6}, {6, 6})})});
    auto sub_device_manager_1 = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    auto sub_device_manager_2 = mesh_device->create_sub_device_manager({sub_device_4, sub_device_3}, k_local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager_1);

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // Run programs on sub-device manager 1
    for (uint32_t i = 0; i < k_num_iters; i++) {
        // Create fresh programs for each iteration
        auto [waiter_program, syncer_program, incrementer_program, global_sem] =
            create_basic_sync_program(mesh_device.get(), sub_device_1, sub_device_2);

        distributed::MeshWorkload waiter_mesh_workload;
        waiter_mesh_workload.add_program(device_range, std::move(waiter_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_mesh_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});

        // Test blocking on one sub-device
        distributed::MeshWorkload syncer_mesh_workload;
        syncer_mesh_workload.add_program(device_range, std::move(syncer_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_mesh_workload, true);

        distributed::MeshWorkload incrementer_mesh_workload;
        incrementer_mesh_workload.add_program(device_range, std::move(incrementer_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_mesh_workload, false);

        mesh_device->reset_sub_device_stall_group();
    }
    distributed::Synchronize(*mesh_device, std::nullopt);

    // Rerun programs on sub-device manager 2
    mesh_device->load_sub_device_manager(sub_device_manager_2);
    for (uint32_t i = 0; i < k_num_iters; i++) {
        // Create fresh programs for each iteration
        auto [waiter_program, syncer_program, incrementer_program, global_sem] =
            create_basic_sync_program(mesh_device.get(), sub_device_1, sub_device_2);

        distributed::MeshWorkload waiter_mesh_workload;
        waiter_mesh_workload.add_program(device_range, std::move(waiter_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_mesh_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});

        // Test blocking on one sub-device
        distributed::MeshWorkload syncer_mesh_workload;
        syncer_mesh_workload.add_program(device_range, std::move(syncer_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_mesh_workload, true);

        distributed::MeshWorkload incrementer_mesh_workload;
        incrementer_mesh_workload.add_program(device_range, std::move(incrementer_program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_mesh_workload, false);

        mesh_device->reset_sub_device_stall_group();
    }
    distributed::Synchronize(*mesh_device, std::nullopt);
    ReadMeshDeviceProfilerResults(*mesh_device);
}

// Ensure each core in the sub device aware of their own logical coordinate. Same binary used in multiple sub devices.
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixTestSubDeviceMyLogicalCoordinates) {
    auto mesh_device = devices_[0];
    // Make 2 sub devices.
    // origin means top left.
    // for sub_device_1 = 0,0. so relative coordinates are the same as logical.
    // for sub_device_2 = 3,3. so relative coordinates are offset by x=3,y=3 from logical.
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {6, 6})})});

    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager);

    const auto sub_device_1_cores = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    const auto sub_device_2_cores = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{1});

    uint32_t cb_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    std::vector<uint32_t> compile_args{cb_addr};

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // Start kernels on each sub device and verify their coordinates
    Program program_1 = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program_1,
        k_coordinates_kernel_path,
        sub_device_1_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = compile_args});

    Program program_2 = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateKernel(
        program_2,
        k_coordinates_kernel_path,
        sub_device_2_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = compile_args});

    distributed::MeshWorkload mesh_workload_1;
    mesh_workload_1.add_program(device_range, std::move(program_1));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_1, false);

    distributed::MeshWorkload mesh_workload_2;
    mesh_workload_2.add_program(device_range, std::move(program_2));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_2, false);

    distributed::Finish(mesh_device->mesh_command_queue());
    mesh_device->reset_sub_device_stall_group();
    distributed::Synchronize(
        *mesh_device, std::nullopt);  // Ensure this CQ is cleared. Each CQ can only work on 1 sub device

    // Check coordinates
    tt::tt_metal::verify_kernel_coordinates(
        HalProgrammableCoreType::TENSIX, sub_device_1_cores, mesh_device.get(), tt::tt_metal::SubDeviceId{0}, cb_addr);
    tt::tt_metal::verify_kernel_coordinates(
        HalProgrammableCoreType::TENSIX, sub_device_2_cores, mesh_device.get(), tt::tt_metal::SubDeviceId{1}, cb_addr);
}

// Ensure the relative coordinate for the worker is updated correctly when it is used for multiple sub device
// configurations
TEST_F(UnitMeshCQSingleCardProgramFixture, TensixTestSubDeviceMyLogicalCoordinatesSubDeviceSwitch) {
    auto mesh_device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    SubDevice sub_device_3(std::array{CoreRangeSet(CoreRange({4, 4}, {6, 6}))});
    SubDevice sub_device_4(std::array{CoreRangeSet(std::vector{CoreRange({2, 2}, {2, 2}), CoreRange({3, 3}, {3, 3})})});
    std::vector<SubDeviceManagerId> sub_device_managers{
        mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size),
        mesh_device->create_sub_device_manager({sub_device_3, sub_device_4}, k_local_l1_size),
    };

    uint32_t cb_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    std::vector<uint32_t> compile_args{cb_addr};

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    for (int i = 0; i < sub_device_managers.size(); ++i) {
        mesh_device->load_sub_device_manager(sub_device_managers[i]);
        const auto sub_device_cores = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{i});

        Program program = tt::tt_metal::CreateProgram();
        tt::tt_metal::CreateKernel(
            program,
            k_coordinates_kernel_path,
            sub_device_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args});

        distributed::MeshWorkload mesh_workload;
        mesh_workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

        distributed::Finish(mesh_device->mesh_command_queue());
        mesh_device->reset_sub_device_stall_group();
        distributed::Synchronize(
            *mesh_device,
            mesh_device->mesh_command_queue(0));  // Ensure this CQ is cleared. Each CQ can only work on 1 sub device

        // Check coordinates
        tt::tt_metal::verify_kernel_coordinates(
            HalProgrammableCoreType::TENSIX,
            sub_device_cores,
            mesh_device.get(),
            tt::tt_metal::SubDeviceId{i},
            cb_addr);
    }
}

// Test that RTAs will be correctly updated when using the same program on multiple subdevice managers.
TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceProgramReuseRtas) {
    constexpr uint32_t k_num_iters = 5;
    auto mesh_device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(CoreRange({3, 3}, {3, 3}))});
    // Sub device IDs are swapped between the two sub device managers.
    auto sub_device_manager_1 = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    auto sub_device_manager_2 = mesh_device->create_sub_device_manager({sub_device_2, sub_device_1}, k_local_l1_size);

    uint32_t l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    CoreCoord core = {3, 3};
    std::array<uint32_t, 1> unique_runtime_args = {101};
    std::array<uint32_t, 1> common_runtime_args = {201};

    for (size_t i = 0; i < k_num_iters; i++) {
        for (const auto& sub_device_manager : {sub_device_manager_1, sub_device_manager_2}) {
            mesh_device->load_sub_device_manager(sub_device_manager);
            unique_runtime_args[0] += 1;
            common_runtime_args[0] += 2;

            // Create fresh program for each iteration and set runtime args
            auto create_program_with_args = [&]() {
                tt_metal::Program program = tt_metal::CreateProgram();
                tt_metal::KernelHandle add_two_ints_kernel = tt_metal::CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/add_common_and_unique_rta.cpp",
                    core,
                    tt_metal::DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt_metal::NOC::RISCV_0_default,
                        .compile_args = {l1_unreserved_base}});

                tt_metal::SetCommonRuntimeArgs(program, add_two_ints_kernel, common_runtime_args);
                tt_metal::SetRuntimeArgs(program, add_two_ints_kernel, core, unique_runtime_args);
                return program;
            };

            // Enqueue twice to ensure waits are correct.
            distributed::MeshWorkload mesh_workload_1;
            mesh_workload_1.add_program(device_range, create_program_with_args());
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_1, false);

            distributed::MeshWorkload mesh_workload_2;
            mesh_workload_2.add_program(device_range, create_program_with_args());
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_2, false);

            distributed::Synchronize(*mesh_device, std::nullopt);
            std::vector<uint32_t> kernel_result;
            slow_dispatch::ReadFromL1(*mesh_device, core, l1_unreserved_base, sizeof(int), kernel_result);
            EXPECT_EQ(kernel_result[0], unique_runtime_args[0] + common_runtime_args[0]);
        }
    }
}

TEST_F(UnitMeshMultiCQSingleDeviceFixture, TensixTestSubDeviceCQOwnership) {
    auto mesh_device = device_;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(CoreRange({3, 3}, {3, 3}))});
    // Sub device IDs are swapped between the two sub device managers.
    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager);

    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    // Helper functions to create fresh programs each time
    auto create_program_1 = [&]() {
        tt_metal::Program program = tt_metal::CreateProgram();
        // On sub device 1.
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            CoreRangeSet(CoreRange({0, 0}, {2, 2})),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
        return program;
    };

    auto create_program_2 = [&]() {
        tt_metal::Program program = tt_metal::CreateProgram();
        // On sub device 2.
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            CoreRangeSet(CoreRange({3, 3}, {3, 3})),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
        return program;
    };

    distributed::MeshWorkload mesh_workload_1;
    mesh_workload_1.add_program(device_range, create_program_1());

    distributed::MeshWorkload mesh_workload_2;
    mesh_workload_2.add_program(device_range, create_program_2());

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_1, false);
    std::array sub_device_ids_for_event = {SubDeviceId{1}};
    auto early_event = mesh_device->mesh_command_queue(1).enqueue_record_event_to_host(sub_device_ids_for_event);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(1), mesh_workload_2, false);

    // CQ 0 owns sub device 1, CQ 1 owns sub device 2.
    // This should throw because program_1 targets sub device 1 which is owned by CQ 0
    EXPECT_THROW(
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(1), mesh_workload_1, false), std::exception);

    // Finish allows transfering ownership of sub device 1.
    distributed::Finish(mesh_device->mesh_command_queue(0));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(1), mesh_workload_1, false);

    // CQ 1 owns sub devices 1 and 2.
    // This should throw because program_2 targets sub device 2 which is now owned by CQ 1
    EXPECT_THROW(
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_2, false), std::exception);
    // Waiting on an event before the last program was queued does not allow transferring ownership of sub device 2.
    mesh_device->mesh_command_queue(0).enqueue_wait_for_event(early_event);
    EXPECT_THROW(
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_2, false), std::exception);

    // Later event allows transferring ownership of sub device 2 to CQ 0
    auto event1 = mesh_device->mesh_command_queue(1).enqueue_record_event_to_host(sub_device_ids_for_event);
    auto event2 = mesh_device->mesh_command_queue(1).enqueue_record_event_to_host(sub_device_ids_for_event);
    log_info(tt::LogTest, "waiting on event2");
    mesh_device->mesh_command_queue(0).enqueue_wait_for_event(event2);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_2, false);

    distributed::Synchronize(*mesh_device, std::nullopt);

    // Synchronize allows transferring ownership of either subdevice.
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_1, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(1), mesh_workload_2, false);

    distributed::Finish(mesh_device->mesh_command_queue(0));
    distributed::Finish(mesh_device->mesh_command_queue(1));
}

// A trace replay moves the workers' go and launch state by an amount the host computes rather than observes (the
// replay's reset go plus the traced program gos), and the next CQ to take ownership of the sub-device continues from
// that state. Replay on one CQ, hand the sub-device to the other CQ (and back), and count how many times every core
// actually ran: state the new owner gets wrong drops or repeats a go, leaving the count off (or hanging the wait).
TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixTestSubDeviceCQOwnershipAfterTraceReplay) {
    if (this->num_cqs_ != 2) {
        GTEST_SKIP() << "This test must be run with TT_METAL_GTEST_NUM_HW_CQS=2";
    }
    CreateDevice(16384);
    auto mesh_device = device_;

    const CoreRange worker_range({0, 0}, {2, 2});
    SubDevice sub_device(std::array{CoreRangeSet(worker_range)});
    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device}, k_local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager);

    const uint32_t counter_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> zero = {0};
        slow_dispatch::WriteToL1(*mesh_device, core, counter_addr, zero);
    }

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/local_counter_increment.cpp",
        CoreRangeSet(worker_range),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {counter_addr}});
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, std::move(program));

    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);
    uint32_t expected_runs = 0;

    // Compile and run once on CQ 0, then capture a trace of several launches on CQ 0.
    distributed::EnqueueMeshWorkload(cq0, workload, false);
    expected_runs += 1;
    constexpr uint32_t k_programs_in_trace = 3;
    auto trace_id = mesh_device->begin_mesh_trace(cq0);
    for (uint32_t i = 0; i < k_programs_in_trace; i++) {
        distributed::EnqueueMeshWorkload(cq0, workload, false);
    }
    mesh_device->end_mesh_trace(cq0, trace_id);

    // Replay on CQ 0, then hand the sub-device to CQ 1, which continues from the state the replays left.
    constexpr uint32_t k_replays = 2;
    for (uint32_t i = 0; i < k_replays; i++) {
        mesh_device->replay_mesh_trace(cq0, trace_id, false);
        expected_runs += k_programs_in_trace;
    }
    distributed::Finish(cq0);
    for (uint32_t i = 0; i < 2; i++) {
        distributed::EnqueueMeshWorkload(cq1, workload, false);
        expected_runs += 1;
    }
    distributed::Finish(cq1);

    // And back: CQ 0 retakes the sub-device through the replay path, then CQ 1 retakes it right after that replay.
    mesh_device->replay_mesh_trace(cq0, trace_id, false);
    expected_runs += k_programs_in_trace;
    distributed::Finish(cq0);
    distributed::EnqueueMeshWorkload(cq1, workload, false);
    expected_runs += 1;
    distributed::Finish(cq1);

    distributed::Synchronize(*mesh_device, std::nullopt);
    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> counter;
        slow_dispatch::ReadFromL1(*mesh_device, core, counter_addr, sizeof(uint32_t), counter);
        EXPECT_EQ(counter[0], expected_runs) << "core " << core.str();
    }
    mesh_device->release_mesh_trace(trace_id);
}

// A trace captured on one CQ can be replayed on another. Its programs' workers must report done to the replaying CQ:
// reporting to the capturing CQ's dispatcher leaves the replaying CQ waiting for completions that never arrive.
// Capture on CQ 0, replay on CQ 1 and on CQ 0 in turn, and count how many times every core ran.
TEST_F(UnitMeshMultiCQSingleDeviceTraceFixture, TensixTestTraceReplayOnOtherCQ) {
    if (this->num_cqs_ != 2) {
        GTEST_SKIP() << "This test must be run with TT_METAL_GTEST_NUM_HW_CQS=2";
    }
    CreateDevice(16384);
    auto mesh_device = device_;

    const CoreRange worker_range({0, 0}, {2, 2});
    const uint32_t counter_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> zero = {0};
        slow_dispatch::WriteToL1(*mesh_device, core, counter_addr, zero);
    }

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/local_counter_increment.cpp",
        CoreRangeSet(worker_range),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {counter_addr}});
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, std::move(program));

    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);
    uint32_t expected_runs = 0;

    // Compile and run once on CQ 0, then capture a trace of several launches on CQ 0.
    distributed::EnqueueMeshWorkload(cq0, workload, false);
    expected_runs += 1;
    constexpr uint32_t k_programs_in_trace = 3;
    auto trace_id = mesh_device->begin_mesh_trace(cq0);
    for (uint32_t i = 0; i < k_programs_in_trace; i++) {
        distributed::EnqueueMeshWorkload(cq0, workload, false);
    }
    mesh_device->end_mesh_trace(cq0, trace_id);
    distributed::Finish(cq0);

    constexpr uint32_t k_rounds = 2;
    for (uint32_t i = 0; i < k_rounds; i++) {
        mesh_device->replay_mesh_trace(cq1, trace_id, false);
        expected_runs += k_programs_in_trace;
        distributed::Finish(cq1);
        mesh_device->replay_mesh_trace(cq0, trace_id, false);
        expected_runs += k_programs_in_trace;
        distributed::Finish(cq0);
    }

    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> counter;
        slow_dispatch::ReadFromL1(*mesh_device, core, counter_addr, sizeof(uint32_t), counter);
        EXPECT_EQ(counter[0], expected_runs) << "core " << core.str();
    }
    mesh_device->release_mesh_trace(trace_id);
}

// A CQ that takes over a sub-device with an enqueue that sends the workers no go (an empty program) must still get the
// done signals of its next program. Otherwise that program reports done to the previous owner and the CQ's finish
// never completes.
TEST_F(UnitMeshMultiCQSingleDeviceFixture, TensixTestCQOwnershipTakenByEmptyWorkload) {
    auto mesh_device = device_;
    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);

    distributed::MeshWorkload blank_workload;
    blank_workload.add_program(device_range_, create_blank_program(CoreRange({0, 0}, {0, 0})));
    distributed::MeshWorkload empty_workload;
    empty_workload.add_program(device_range_, tt_metal::CreateProgram());

    distributed::EnqueueMeshWorkload(cq0, blank_workload, false);
    distributed::Finish(cq0);
    distributed::EnqueueMeshWorkload(cq1, empty_workload, false);
    distributed::EnqueueMeshWorkload(cq1, blank_workload, false);
    distributed::Finish(cq1);
}

// A CQ that takes over a sub-device by waiting on the previous owner's event must not redirect the done signals of the
// previous owner's still-running programs: a worker still running the previous owner's program would report done to
// the new owner, and the previous owner's event would never complete. Block one core on CQ 0, hand the sub-device to
// CQ 1 through an event, and give CQ 1 time to run ahead before releasing the core.
TEST_F(UnitMeshMultiCQSingleDeviceFixture, TensixTestCQOwnershipHandoffBehindRunningProgram) {
    auto mesh_device = device_;
    auto& cq0 = mesh_device->mesh_command_queue(0);
    auto& cq1 = mesh_device->mesh_command_queue(1);

    const CoreCoord blocked_core{0, 0};
    auto semaphore = CreateGlobalSemaphore(*mesh_device, CoreRangeSet(CoreRange(blocked_core, blocked_core)), 0);
    tt_metal::Program program = tt_metal::CreateProgram();
    auto kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        CoreRangeSet(CoreRange(blocked_core, blocked_core)),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    tt_metal::SetRuntimeArgs(program, kernel, blocked_core, {static_cast<uint32_t>(semaphore.address())});
    distributed::MeshWorkload blocked_workload;
    blocked_workload.add_program(device_range_, std::move(program));
    distributed::MeshWorkload blank_workload;
    blank_workload.add_program(device_range_, create_blank_program(CoreRange({1, 1}, {1, 1})));

    distributed::EnqueueMeshWorkload(cq0, blocked_workload, false);
    auto event = cq0.enqueue_record_event_to_host();
    cq1.enqueue_wait_for_event(event);
    distributed::EnqueueMeshWorkload(cq1, blank_workload, false);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(distributed::EventQuery(event));

    std::vector<uint32_t> release = {1};
    slow_dispatch::WriteToL1(*mesh_device, blocked_core, semaphore.address(), release);
    distributed::EventSynchronize(event);
    distributed::Finish(cq1);
    distributed::Finish(cq0);
}

// Switching sub-device managers moves cores the outgoing manager left unassigned into a sub-device whose go state the
// switch's reset just touched. They must not take that reset as new work: acking it a second time would leave extra
// counts in the sub-device's completion counter, letting a later wait for workers pass before they finish. Block one
// core's program on a semaphore and check that an event recorded behind it does not complete until the core is
// released.
TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceSwitchToDefaultNoStrayResetAcks) {
    auto mesh_device = devices_[0];
    auto& cq = mesh_device->mesh_command_queue();

    // Cores outside {0,0}-{2,2} are unassigned under this manager.
    auto sub_device_manager = mesh_device->create_sub_device_manager(
        {SubDevice(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))})}, k_local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager);
    mesh_device->clear_loaded_sub_device_manager();

    const CoreCoord blocked_core{0, 0};
    auto semaphore = CreateGlobalSemaphore(*mesh_device, CoreRangeSet(CoreRange(blocked_core, blocked_core)), 0);
    tt_metal::Program program = tt_metal::CreateProgram();
    auto kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/syncer.cpp",
        CoreRangeSet(CoreRange(blocked_core, blocked_core)),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
    tt_metal::SetRuntimeArgs(program, kernel, blocked_core, {static_cast<uint32_t>(semaphore.address())});
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, std::move(program));

    distributed::EnqueueMeshWorkload(cq, workload, false);
    auto event = cq.enqueue_record_event_to_host();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_FALSE(distributed::EventQuery(event));

    std::vector<uint32_t> release = {1};
    slow_dispatch::WriteToL1(*mesh_device, blocked_core, semaphore.address(), release);
    distributed::EventSynchronize(event);
    distributed::Finish(cq);
    mesh_device->remove_sub_device_manager(sub_device_manager);
}

// Single CQ: switching sub-device managers right behind a non-blocking trace replay must not disturb the replay. The
// switch resets every worker's go state, and with one CQ nothing drains on the host first, so the reset has to be
// ordered behind the replay on the device. The workers also move go rows (sub-device 0 -> sub-device 1) across the
// switch. A reset that overtakes the replay is not reliably visible in the final count (the stray runs it causes can
// replay already-consumed launch messages), so the test also samples a worker's go messages from the host while the
// slow replay is still running: before the replay's programs have all run, none may hold a reset.
TEST_F(UnitMeshCQTraceFixture, TensixTestSubDeviceSwitchBehindTraceReplay) {
    CreateDevices(16384);
    auto mesh_device = this->devices_[0];
    auto& cq = mesh_device->mesh_command_queue();

    const CoreRange worker_range({0, 0}, {2, 2});
    const CoreRange other_range({3, 3}, {3, 3});
    auto manager_a =
        mesh_device->create_sub_device_manager({SubDevice(std::array{CoreRangeSet(worker_range)})}, k_local_l1_size);
    auto manager_b = mesh_device->create_sub_device_manager(
        {SubDevice(std::array{CoreRangeSet(other_range)}), SubDevice(std::array{CoreRangeSet(worker_range)})},
        k_local_l1_size);
    mesh_device->load_sub_device_manager(manager_a);

    const uint32_t counter_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> zero = {0};
        slow_dispatch::WriteToL1(*mesh_device, core, counter_addr, zero);
    }

    // A counter on RISCV_1, optionally alongside a delay on RISCV_0 so each launch stays on the workers a while.
    auto make_counter_workload = [&](std::optional<uint32_t> delay_iters) {
        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/local_counter_increment.cpp",
            CoreRangeSet(worker_range),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {counter_addr}});
        if (delay_iters.has_value()) {
            auto delay_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/delay.cpp",
                CoreRangeSet(worker_range),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            std::array<uint32_t, 1> delay_rt_args = {*delay_iters};
            tt_metal::SetRuntimeArgs(program, delay_kernel, CoreRangeSet(worker_range), delay_rt_args);
        }
        distributed::MeshWorkload workload;
        workload.add_program(device_range_, std::move(program));
        return workload;
    };
    constexpr uint32_t k_delay_iters = 5000000;  // long enough that the host samples the go message mid-replay
    auto slow_workload = make_counter_workload(k_delay_iters);
    auto workload_b = make_counter_workload(std::nullopt);
    uint32_t expected_runs = 0;

    // Compile under manager A, then capture several launches of the slow counter.
    distributed::EnqueueMeshWorkload(cq, slow_workload, false);
    expected_runs += 1;
    constexpr uint32_t k_programs_in_trace = 4;
    auto trace_id = mesh_device->begin_mesh_trace(cq);
    for (uint32_t i = 0; i < k_programs_in_trace; i++) {
        distributed::EnqueueMeshWorkload(cq, slow_workload, false);
    }
    mesh_device->end_mesh_trace(cq, trace_id);

    // Under manager A the workers are in sub-device 0, whose go ring is the first one at GO_MSG. The switch's reset
    // goes to the slot of the host's launch message write pointer, so the whole ring is probed.
    const CoreCoord probe_core = worker_range.start_coord;
    const auto& hal = MetalContext::instance().hal();
    const uint32_t go_ring_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    auto go_ring = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX).create<dev_msgs::go_ring_t>();
    uint32_t mid_replay_samples = 0;

    constexpr uint32_t k_iters = 3;
    for (uint32_t i = 0; i < k_iters; i++) {
        mesh_device->replay_mesh_trace(cq, trace_id, false);
        expected_runs += k_programs_in_trace;
        mesh_device->load_sub_device_manager(manager_b);
        // Read the go messages before the counter: the counter only grows, so a short count means the replay was
        // still running when the go messages were read.
        std::vector<uint32_t> go_words;
        std::vector<uint32_t> counter;
        slow_dispatch::ReadFromL1(*mesh_device, probe_core, go_ring_addr, go_ring.size(), go_words);
        slow_dispatch::ReadFromL1(*mesh_device, probe_core, counter_addr, sizeof(uint32_t), counter);
        std::memcpy(go_ring.data(), go_words.data(), go_ring.size());
        if (counter[0] < expected_runs) {
            mid_replay_samples++;
            auto slots = go_ring.view().slot();
            for (uint32_t slot = 0; slot < slots.size(); slot++) {
                const uint32_t signal = slots[slot].signal();
                EXPECT_TRUE(
                    signal != dev_msgs::RUN_MSG_RESET_READ_PTR && signal != dev_msgs::RUN_MSG_RESET_READ_PTR_FROM_HOST)
                    << "go message in slot " << slot << " reset while the replay was still running (signal " << signal
                    << ", counter " << counter[0] << " of " << expected_runs << ")";
            }
        }
        distributed::EnqueueMeshWorkload(cq, workload_b, false);
        expected_runs += 1;
        // The trace belongs to the manager it was captured under.
        mesh_device->load_sub_device_manager(manager_a);
    }
    distributed::Finish(cq);
    if (mid_replay_samples == 0) {
        log_warning(tt::LogTest, "The go message was never sampled mid-replay; the ordering check did not run");
    }

    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> counter;
        slow_dispatch::ReadFromL1(*mesh_device, core, counter_addr, sizeof(uint32_t), counter);
        EXPECT_EQ(counter[0], expected_runs) << "core " << core.str();
    }
    mesh_device->release_mesh_trace(trace_id);
}

// A sub-device manager switch resets the workers' launch message slot, so whatever slot the workers are on when the
// switch comes, they and the host must agree afterwards. Switch with the workers on every slot of the launch ring (and
// past its end), moving them between sub-devices each time, and count how many times every core ran.
TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceSwitchFromEveryLaunchSlot) {
    auto mesh_device = devices_[0];
    auto& cq = mesh_device->mesh_command_queue();

    const CoreRange worker_range({0, 0}, {2, 2});
    const CoreRange other_range({3, 3}, {3, 3});
    auto manager_a =
        mesh_device->create_sub_device_manager({SubDevice(std::array{CoreRangeSet(worker_range)})}, k_local_l1_size);
    auto manager_b = mesh_device->create_sub_device_manager(
        {SubDevice(std::array{CoreRangeSet(other_range)}), SubDevice(std::array{CoreRangeSet(worker_range)})},
        k_local_l1_size);
    mesh_device->load_sub_device_manager(manager_a);

    const uint32_t counter_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> zero = {0};
        slow_dispatch::WriteToL1(*mesh_device, core, counter_addr, zero);
    }

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/local_counter_increment.cpp",
        CoreRangeSet(worker_range),
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {counter_addr}});
    distributed::MeshWorkload workload;
    workload.add_program(device_range_, std::move(program));

    // One more than the launch message ring depth, so the last round also wraps the ring.
    constexpr uint32_t k_max_programs_before_switch = 9;
    uint32_t expected_runs = 0;
    for (uint32_t programs = 0; programs <= k_max_programs_before_switch; programs++) {
        for (uint32_t i = 0; i < programs; i++) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
            expected_runs++;
        }
        mesh_device->load_sub_device_manager(manager_b);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        expected_runs++;
        mesh_device->load_sub_device_manager(manager_a);
    }
    distributed::Finish(cq);

    for (const CoreCoord& core : worker_range) {
        std::vector<uint32_t> counter;
        slow_dispatch::ReadFromL1(*mesh_device, core, counter_addr, sizeof(uint32_t), counter);
        EXPECT_EQ(counter[0], expected_runs) << "core " << core.str();
    }
    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(manager_a);
    mesh_device->remove_sub_device_manager(manager_b);
}

}  // namespace tt::tt_metal
