// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/data_types.hpp>
#include "dispatch_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
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

namespace tt::tt_metal {

constexpr uint32_t k_local_l1_size = 3200;
const std::string k_coordinates_kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/read_my_coordinates.cpp";

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceCBAllocation) {
    auto mesh_device = devices_[0];
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});
    SubDevice sub_device_1(std::array{sharded_cores_1});
    auto sub_device_manager_1 = mesh_device->create_sub_device_manager({sub_device_1}, k_local_l1_size);
    DeviceAddr l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    DeviceAddr l1_max_size = mesh_device->get_devices()[0]->l1_size_per_core();
    DeviceAddr l1_total_size = l1_max_size - l1_unreserved_base;
    mesh_device->load_sub_device_manager(sub_device_manager_1);
    uint32_t global_buffer_size = l1_total_size - k_local_l1_size * 2;
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

    program.impl().allocate_circular_buffers(mesh_device.get());
    program.impl().validate_circular_buffer_region(mesh_device.get());
    UpdateCircularBufferTotalSize(program, cb_src0, k_local_l1_size * 3);
    program.impl().allocate_circular_buffers(mesh_device.get());
    EXPECT_THROW(program.impl().validate_circular_buffer_region(mesh_device.get()), std::exception);
    global_buffer.reset();
    program.impl().validate_circular_buffer_region(mesh_device.get());
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
    UpdateCircularBufferTotalSize(program, cb_src0, k_local_l1_size / 4);
    program.impl().allocate_circular_buffers(mesh_device.get());
    program.impl().validate_circular_buffer_region(mesh_device.get());
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
    distributed::MeshWorkload mesh_workload = distributed::CreateMeshWorkload();
    distributed::MeshCoordinate zero_coord = distributed::MeshCoordinate::zero_coordinate(device->shape().dims());
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::AddProgramToMeshWorkload(mesh_workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
    device->set_sub_device_stall_group(sub_device_ids_to_block);

    auto buffer_1 = distributed::MeshBuffer::create(replicated_config_1, local_config_1, device);

    // Test blocking synchronize doesn't stall
    distributed::Synchronize(device, std::nullopt);

    // Test blocking write buffer doesn't stall
    distributed::EnqueueWriteMeshBuffer(device->mesh_command_queue(), buffer_1, input_1, true);

    // Test record event won't cause a stall

    auto event = distributed::EnqueueRecordEventToHost(device->mesh_command_queue());
    distributed::Synchronize(device, std::nullopt);

    // Test blocking read buffer doesn't stall
    std::vector<uint32_t> output_1;
    distributed::ReadShard(device->mesh_command_queue(), output_1, buffer_1, zero_coord, true);
    EXPECT_EQ(input_1, output_1);
    auto input_1_it = input_1.begin();
    for (const auto& physical_core : physical_cores_1) {
        auto readback = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            device->get_devices()[0]->id(), physical_core, buffer_1->address(), page_size_1);
        EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
        input_1_it += page_size_1 / sizeof(uint32_t);
    }
    auto sem_addr = global_semaphore.address();
    auto physical_syncer_core = device->worker_core_from_logical_core(syncer_core);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        device->get_devices()[0]->id(), physical_syncer_core, std::vector<uint32_t>{1}, sem_addr);

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

        distributed::MeshWorkload waiter_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(waiter_mesh_workload, std::move(waiter_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_mesh_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});

        // Test blocking on one sub-device
        distributed::MeshWorkload syncer_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(syncer_mesh_workload, std::move(syncer_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_mesh_workload, true);

        distributed::MeshWorkload incrementer_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(incrementer_mesh_workload, std::move(incrementer_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_mesh_workload, false);

        mesh_device->reset_sub_device_stall_group();
    }
    distributed::Synchronize(mesh_device.get(), std::nullopt);
    detail::ReadDeviceProfilerResults(mesh_device->get_devices()[0]);
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

        distributed::MeshWorkload waiter_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(waiter_mesh_workload, std::move(waiter_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_mesh_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{0}}});

        // Test blocking on one sub-device
        distributed::MeshWorkload syncer_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(syncer_mesh_workload, std::move(syncer_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_mesh_workload, true);

        distributed::MeshWorkload incrementer_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(incrementer_mesh_workload, std::move(incrementer_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_mesh_workload, false);

        mesh_device->reset_sub_device_stall_group();
    }
    distributed::Synchronize(mesh_device.get(), std::nullopt);

    // Rerun programs on sub-device manager 2
    mesh_device->load_sub_device_manager(sub_device_manager_2);
    for (uint32_t i = 0; i < k_num_iters; i++) {
        // Create fresh programs for each iteration
        auto [waiter_program, syncer_program, incrementer_program, global_sem] =
            create_basic_sync_program(mesh_device.get(), sub_device_1, sub_device_2);

        distributed::MeshWorkload waiter_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(waiter_mesh_workload, std::move(waiter_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), waiter_mesh_workload, false);

        mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});

        // Test blocking on one sub-device
        distributed::MeshWorkload syncer_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(syncer_mesh_workload, std::move(syncer_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), syncer_mesh_workload, true);

        distributed::MeshWorkload incrementer_mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(incrementer_mesh_workload, std::move(incrementer_program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), incrementer_mesh_workload, false);

        mesh_device->reset_sub_device_stall_group();
    }
    distributed::Synchronize(mesh_device.get(), std::nullopt);
    detail::ReadDeviceProfilerResults(mesh_device->get_devices()[0]);
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

    distributed::MeshWorkload mesh_workload_1 = distributed::CreateMeshWorkload();
    distributed::AddProgramToMeshWorkload(mesh_workload_1, std::move(program_1), device_range);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_1, false);

    distributed::MeshWorkload mesh_workload_2 = distributed::CreateMeshWorkload();
    distributed::AddProgramToMeshWorkload(mesh_workload_2, std::move(program_2), device_range);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_2, false);

    distributed::Finish(mesh_device->mesh_command_queue());
    mesh_device->reset_sub_device_stall_group();
    distributed::Synchronize(
        mesh_device.get(), std::nullopt);  // Ensure this CQ is cleared. Each CQ can only work on 1 sub device

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

        distributed::MeshWorkload mesh_workload = distributed::CreateMeshWorkload();
        distributed::AddProgramToMeshWorkload(mesh_workload, std::move(program), device_range);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

        distributed::Finish(mesh_device->mesh_command_queue());
        mesh_device->reset_sub_device_stall_group();
        distributed::Synchronize(
            mesh_device.get(), 0);  // Ensure this CQ is cleared. Each CQ can only work on 1 sub device

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
        for (auto& sub_device_manager : {sub_device_manager_1, sub_device_manager_2}) {
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
            distributed::MeshWorkload mesh_workload_1 = distributed::CreateMeshWorkload();
            distributed::AddProgramToMeshWorkload(mesh_workload_1, create_program_with_args(), device_range);
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_1, false);

            distributed::MeshWorkload mesh_workload_2 = distributed::CreateMeshWorkload();
            distributed::AddProgramToMeshWorkload(mesh_workload_2, create_program_with_args(), device_range);
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload_2, false);

            distributed::Synchronize(mesh_device.get(), std::nullopt);
            std::vector<uint32_t> kernel_result;
            tt_metal::detail::ReadFromDeviceL1(
                mesh_device->get_devices()[0], core, l1_unreserved_base, sizeof(int), kernel_result);
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
            "tt_metal/kernels/dataflow/blank.cpp",
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
            "tt_metal/kernels/dataflow/blank.cpp",
            CoreRangeSet(CoreRange({3, 3}, {3, 3})),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
        return program;
    };

    distributed::MeshWorkload mesh_workload_1 = distributed::CreateMeshWorkload();
    distributed::AddProgramToMeshWorkload(mesh_workload_1, create_program_1(), device_range);

    distributed::MeshWorkload mesh_workload_2 = distributed::CreateMeshWorkload();
    distributed::AddProgramToMeshWorkload(mesh_workload_2, create_program_2(), device_range);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_1, false);
    std::array sub_device_ids_for_event = {SubDeviceId{1}};
    auto early_event =
        distributed::EnqueueRecordEventToHost(mesh_device->mesh_command_queue(1), sub_device_ids_for_event);
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
    distributed::EnqueueWaitForEvent(mesh_device->mesh_command_queue(0), early_event);
    EXPECT_THROW(
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_2, false), std::exception);

    // Later event allows transferring ownership of sub device 2 to CQ 0
    auto event1 = distributed::EnqueueRecordEventToHost(mesh_device->mesh_command_queue(1), sub_device_ids_for_event);
    auto event2 = distributed::EnqueueRecordEventToHost(mesh_device->mesh_command_queue(1), sub_device_ids_for_event);
    log_info(tt::LogTest, "waiting on event2");
    distributed::EnqueueWaitForEvent(mesh_device->mesh_command_queue(0), event2);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_2, false);

    distributed::Synchronize(mesh_device.get(), std::nullopt);

    // Synchronize allows transferring ownership of either subdevice.
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(0), mesh_workload_1, false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(1), mesh_workload_2, false);

    distributed::Finish(mesh_device->mesh_command_queue(0));
    distributed::Finish(mesh_device->mesh_command_queue(1));
}

}  // namespace tt::tt_metal
